// Example Compute Shader

uniform float uTime;
uniform float uDelta;
uniform vec4 uNoiseFactor;
uniform float uVelDamping;
uniform float uMaxSpeed;
uniform float uAbsMaxSteerForce;
uniform int uNumAttractors;
uniform samplerBuffer aAttractorTransRad;
uniform float uAttractionForce;
uniform int uAgentCount;

#define PI 3.14159265358979323846
#define HALFPI 1.57079632679489661923
#define TWOPI 6.283185307179586
#define deg2rad 0.01745329251

#define IN_INIT_POS 0
#define IN_POSITION_AND_MASS_BUFFER 1
#define IN_VELOCITY_AND_SIZE_BUFFER 2
#define IN_DIRECTION_AND_HEALTH_BUFFER 3
#define IN_DNA_BUFFER 4
#define IN_DRAW_AGENTS_BUFFER 5

layout (local_size_x = 8, local_size_y = 8) in;

// CLASS AGENT
struct Agent
{
	vec3 position;
	vec3 direction;
	vec3 speed;
	float maxSpeed;
	vec3 acceleration;
	float mass;
	float size;
	vec4 DNA;
	float health;
	int espece;
};

vec3 snoiseVec3( vec3 x )
{
  float s  = TDSimplexNoise(vec3( x ));
  float s1 = TDSimplexNoise(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
  float s2 = TDSimplexNoise(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
  vec3 c = vec3( s , s1 , s2 );
  return c;

}

vec3 curlNoise( vec3 p )
{
  const float e = .1;
  vec3 dx = vec3( e   , 0.0 , 0.0 );
  vec3 dy = vec3( 0.0 , e   , 0.0 );
  vec3 dz = vec3( 0.0 , 0.0 , e   );

  vec3 p_x0 = snoiseVec3( p - dx );
  vec3 p_x1 = snoiseVec3( p + dx );
  vec3 p_y0 = snoiseVec3( p - dy );
  vec3 p_y1 = snoiseVec3( p + dy );
  vec3 p_z0 = snoiseVec3( p - dz );
  vec3 p_z1 = snoiseVec3( p + dz );

  float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
  float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
  float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

  const float divisor = 1.0 / ( 2.0 * e );
  return normalize( vec3( x , y , z ) * divisor );

}

float parabola( float x, float k )
{
    return pow( 4.0*x*(1.0-x), k );
}

vec3 clampVector(vec3 v, float theMax)
{

	float speed = length(v);
	if (speed < .0001) {
		return vec3(0.);
	}
	vec3 n = v / speed;

	speed = clamp(speed, 0., theMax);

	return n*speed;
}

float map(float value, float inLo, float inHi, float outLo, float outHi)
{
	return outLo + (value - inLo) * (outHi - outLo) / (inHi - inLo);
}
// AGENT UPDATE AND CHECKING

int Index ()
{
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 wh = ivec2(uTD2DInfos[0].res.zw); // uResolution
	int index = (coord.x + (coord.y * wh.x));
	return index;
}

bool Active ()
{
	return Index() < uAgentCount;
}

bool Alive(Agent a)
{
	return a.health > 0;
}

void ApplyForce(inout Agent a, vec3 force) // inout is like a pointer
{
	a.acceleration += force / a.mass;
}

void SeekTarget(inout Agent a, vec3 target, float mag, float rad)
{
	vec3 desired = target - a.position;// direction
	float dist = length(desired); 
	desired = normalize(desired); // use only the direction of the vector

	if(dist < rad * 1.2)
	{
		float m = map(dist, 0, rad, 0, uMaxSpeed);
		desired *= m;
	}
	else if(dist < rad * 3)
	{
		desired *= uMaxSpeed;
		vec3 steer = desired - a.speed;
		steer = clamp(steer, vec3(-uAbsMaxSteerForce), vec3(uAbsMaxSteerForce));

		ApplyForce(a, steer * mag);
	}
	else
	{
		return;
	}

	
}

void Attractors(inout Agent a)
{
	for(int i = 0; i < uNumAttractors; i ++)
	{
		vec4 attractor = texelFetch(aAttractorTransRad, i);
		SeekTarget(a, attractor.xyz, uAttractionForce, attractor.w);
	}
}

void Update(inout Agent agent, vec4 initial) // inout is like a pointer
{
	// POSITION rgb AND MASS a
	agent.health -= 0.001;
	if(agent.health <= 0.001)
	{
		agent.health = initial.a;
		agent.position = initial.rgb;
	}
	agent.position += agent.speed;
	
	// VELOCITY rgb AND SIZE a
	
	float size = parabola(1.0 - agent.health, 1.0);

	agent.size = size;

	agent.speed += agent.acceleration;
	agent.speed *= uVelDamping;
	agent.position += agent.speed;
	agent.speed = clampVector(agent.speed, 2.0);
	agent.acceleration = vec3(0.0);
}

Agent Read(ivec2 coord)
{
	Agent a;
	a.acceleration = vec3(0.0);
	vec4 posMass  = texelFetch(sTD2DInputs[IN_POSITION_AND_MASS_BUFFER], coord, 0);
	vec4 velSize = texelFetch(sTD2DInputs[IN_VELOCITY_AND_SIZE_BUFFER],coord, 0);
	vec4 dirHealth = texelFetch(sTD2DInputs[IN_DIRECTION_AND_HEALTH_BUFFER], coord, 0);
	vec4 dna = texelFetch(sTD2DInputs[IN_DNA_BUFFER], coord, 0);

	a.position = posMass.rgb;
	a.mass = posMass.a;
	if(a.mass <= 0.1){a.mass = 0.2;}
	a.speed = velSize.rgb;
	a.health = dirHealth.r;
	a.direction = dirHealth.gba;
	a.DNA = dna;

	return a;
}

void Write (Agent a, ivec2 coord)
{
	imageStore(mTDComputeOutputs[IN_POSITION_AND_MASS_BUFFER], coord, TDOutputSwizzle(vec4(a.position, a.mass)));

	imageStore(mTDComputeOutputs[IN_VELOCITY_AND_SIZE_BUFFER], coord, TDOutputSwizzle(vec4(a.speed, a.size)));

	imageStore(mTDComputeOutputs[IN_DIRECTION_AND_HEALTH_BUFFER], coord, TDOutputSwizzle(vec4(a.health, a.direction)));

	imageStore(mTDComputeOutputs[IN_DNA_BUFFER], coord, TDOutputSwizzle(a.DNA));
}

void main()
{
	Agent agent;

	// READ AGENT DATA
	agent = Read(ivec2(gl_GlobalInvocationID.xy));

	// READ GENERAL DATA	
	vec4 i = texelFetch(sTD2DInputs[IN_INIT_POS], ivec2(gl_GlobalInvocationID.xy), 0);
	vec4 draw = texelFetch(sTD2DInputs[IN_DRAW_AGENTS_BUFFER], ivec2(gl_GlobalInvocationID.xy), 0);

	// UPDATE AGENT DATA
	vec3 noise = uDelta * uNoiseFactor.x * curlNoise(agent.position * uNoiseFactor.y + uTime * uNoiseFactor.z);
	
	ApplyForce(agent, noise * uNoiseFactor.w);
	Attractors(agent);
	Update(agent, i);	

	// WRITE AGENT DATA
	Write(agent,  ivec2(gl_GlobalInvocationID.xy));

	// DRAW AGENTS rgba
	imageStore(mTDComputeOutputs[IN_DRAW_AGENTS_BUFFER], ivec2(gl_GlobalInvocationID.xy), TDOutputSwizzle(draw));
}
