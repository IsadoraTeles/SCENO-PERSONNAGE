
uniform float uTime;
uniform vec2 uRes;                      // its a square with x and y bounds of 512px (uRes.x) and the z bound is two (uRes.y)
uniform vec2 uCounts; 					// uAgentCount, uFoodCount
uniform int uResetTextureAndAgents;
uniform vec3 uSteerConfigs; 			// uVelDamping, uMaxSpeed, uAbsMaxSteerForce
uniform int uAttractionConfigs;			// uAttractionForce
uniform ivec3 uRangeNeighbors; 			// uRangeGravitation, uRangeFlock, uRangeAnts
uniform float uG;
uniform vec2 uMinMaxDistandCroudMin; 	// uCroudMinDist, uCroudMaxDist
uniform vec2 uConfigsAnts; 				// uSensAngleAnts / DNA.g, // uMaxTrailLimits / DNA.r
uniform vec3 uFlockConfigs;	 			// uAlign / DNA.r, uCohesion / DNA.g, uSeparation / DNA.b
uniform vec3 uSeekSettings; 			// uRad, uAngleFlock, seekFoodForce
uniform float uDecayFactorAnts;			// DNA.b
uniform vec4 uNoiseFactor;				// DNA.r, DNA.g, DNA.b
uniform float uDelta;
uniform int uRangeCroud;
uniform float uMutationRate;
uniform float uReproductionChance;
uniform int uNumAttractors;
uniform int uDNA;
uniform float uAvoidDistance;

uniform samplerBuffer sFoodPositions;

#define PI 3.14159265358979323846
#define HALFPI 1.57079632679489661923
#define TWOPI 6.283185307179586
#define deg2rad 0.01745329251

#define IN_INIT_POS 0
#define IN_POSITION_AND_MASS_BUFFER 1
#define IN_VELOCITY_AND_SIZE_BUFFER 2
#define IN_DIRECTION_AND_HEALTH_BUFFER 3
#define IN_DNA_BUFFER 4
#define IN_COLOR_AGENTS_BUFFER 5
#define IN_STATE_BUFFER 6
#define IN_DRAW_FOOD_BUFFER 7
#define IN_DRAW_MASS_COORD_BUFFER 8
#define IN_DRAW_SPEED_HEALTH_BUFFER 9
#define IN_TRAILS_BUFFER_IN 10
#define IN_TRAILS_BUFFER_OUT 0
#define IN_ATTRACTORS_BUFFER 11

layout (local_size_x = 8, local_size_y = 8) in;

// CLASS AGENT
struct Agent
{
	vec3 position;
	vec3 direction;
    vec4 color;
	vec3 speed;
	float maxSpeed;
	float minSpeed;
	vec3 acceleration;
	float mass;
	float size;
	vec4 DNA;
	vec4 DNAScaled;
	float health;
	int espece;
    ivec2 myCoord;
	int crouded;
	vec3 foodMemory[5]; // Memory of food locations
	int memoryIndex; // Index for the next memory to overwrite
	vec3 rememberedFoodLocation;
	
};

struct Food
{
	vec3 position;
	vec4 foodColor;
	float alive;
	int index;
};

// UTILITIES FUNCTIONS

float random (vec2 st) 
{
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float random2(float seed)
{
    float value = fract(sin(seed) * 43758.5453);
    return value;
}

float randomRange(vec2 st, float min, float max) 
{
    return mix(min, max, fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123));
}

vec4 randomVec4Range(ivec2 posOnTexture, float uTime, float min, float max) 
{
    return vec4(
        randomRange(posOnTexture + sin(uTime * .49), min, max),
        randomRange(posOnTexture + cos(uTime * .899), min, max),
        randomRange(posOnTexture * sin(uTime * .0979), min, max),
        randomRange(posOnTexture + sin(uTime * 0.265), min, max)
    );
}

vec3 randomVec3Range(ivec2 posOnTexture, float uTime, float min, float max) 
{
    return vec3(
        randomRange(posOnTexture + sin(uTime * .49), min, max),
        randomRange(posOnTexture + cos(uTime * .029), min, max),
        randomRange(posOnTexture + sin(uTime * .2279), min, max)
    );
}

float gaussianRand(vec2 seed) 
{
    // Box-Muller transformation
    float u = random(seed);
    float v = random(seed + 1234.5678); // an arbitrary number to change the seed
    float x = sqrt(-2.0 * log(u)) * cos(2.0 * PI * v);
    return x * 0.5 + 0.5; // transform from N(0, 1) to range [0, 1]
}

vec3 normx(vec3 a)
{
	return a == vec3(0.) ? a : normalize(a);
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

float Map(float value, float inLo, float inHi, float outLo, float outHi) 
{
    if(inHi == inLo) // prevent division by zero
    {
        // Decide what to do when inHi and inLo are equal (the input range is zero).
        // For instance, return the midpoint of the output range:
        return (outHi + outLo) / 2.0;
    }
    return outLo + (value - inLo) * (outHi - outLo) / (inHi - inLo);
}

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

vec3 constrainVector(vec3 v, float minVal, float maxVal) 
{
    return vec3(clamp(v.x, minVal, maxVal),
                clamp(v.y, minVal, maxVal),
                clamp(v.z, minVal, maxVal));
}

vec3 map_vector(vec3 vector, float min_val, float max_val) {
    float vector_range = max(vector.r, max(vector.g, vector.b)) - min(vector.r, min(vector.g, vector.b));
    vec3 mapped_vector = (vector - vec3(min(vector.r, min(vector.g, vector.b)))) / vector_range * (max_val - min_val) + vec3(min_val);
    return mapped_vector;
}

// AGENT UPDATE AND CHECKING

int Index ()
{
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 wh = ivec2(uTD2DInfos[0].res.zw); // uResolution
	int index = (coord.x + (coord.y * wh.x));
	return index;
}

bool ActiveAgent ()
{
	return Index() < uCounts.x;
}

bool AliveAgent(Agent a)
{
	return a.health > 0;
}

// READING TEXTURES

Agent ReadAgent(ivec2 coord)
{
	Agent a;
	a.acceleration = vec3(0.0);
	vec4 posMass  = texelFetch(sTD2DInputs[IN_POSITION_AND_MASS_BUFFER], coord, 0);
	vec4 velSize = texelFetch(sTD2DInputs[IN_VELOCITY_AND_SIZE_BUFFER],coord, 0);
	vec4 dirHealth = texelFetch(sTD2DInputs[IN_DIRECTION_AND_HEALTH_BUFFER], coord, 0);
	vec4 dna = texelFetch(sTD2DInputs[IN_DNA_BUFFER], coord, 0);
    ivec2 agentCoord = coord;
	
	vec4 stateAndColor = texelFetch(sTD2DInputs[IN_STATE_BUFFER], coord, 0);
    int agentCrouded = int(stateAndColor.r);

	a.color.xyz = stateAndColor.gba;
	a.position = posMass.rgb;
	a.mass = posMass.a;
	if(a.mass <= 0.1){a.mass = 0.2;}
	a.speed = velSize.rgb;
	a.health = dirHealth.a;
	a.direction = dirHealth.rgb;
	a.DNA = dna;
    a.myCoord = agentCoord;
	a.crouded = agentCrouded;
	
	a.maxSpeed = a.DNA.r * 2 + 0.5; // Adding 0.5 to make sure maxSpeed is always greater than 0.5
	a.minSpeed = a.DNA.r + 0.3; // Reducing the multiplication factor to make sure minSpeed is always less than maxSpeed


	if(a.DNA.a < 0.3)
	{
		a.espece = 1;
		a.color = vec4(1.0, 0, 0, a.health);
	}
	else if (a.DNA.a < 0.7)
	{
		a.espece = 2;
		a.color = vec4(0.7, 0, 1.0, a.health);
	}
	else
	{
		a.espece = 3;
		a.color = vec4(0.4, 1, stateAndColor.a, a.health);
	}
	return a;
}

void ReadFoods(inout Food foods[7])
{
	for(int i = 0; i < uCounts.y; i++)
	{
		vec4 foodInfo = texelFetch(sFoodPositions, i);
		foods[i].position.xyz =  foodInfo.rgb;
		foods[i].foodColor =  foodInfo;
		foods[i].index = i;
	}
}

Agent GenerateEmptyAgent()
{
	Agent agent;
	agent.position = vec3(-999999);
	agent.speed = vec3(0.0);
	agent.acceleration = vec3(0.0);
	agent.size = 0.0;
	agent.mass = 0.0;
	agent.maxSpeed = 0.0;
	agent.minSpeed = 0.0;
	agent.direction = vec3(0,0,0);
	agent.health = 0.0;
	agent.DNA = vec4(0.0);
	agent.color = vec4(0.0);
	agent.espece = 0;
    agent.myCoord = ivec2(-9999);
	agent.crouded = 0;

	return agent;
}

// RESET AGENTS

Agent ResetAgents(ivec2 posOnTexture)
{
	vec3 posRandom = randomVec3Range(posOnTexture, uTime, 0.0, 1.0);
	vec3 velRandom;
    velRandom.x = gaussianRand(posOnTexture + vec2(uTime, 0)) * 2.0 - 1.0; // generate values from -1 to 1
    velRandom.y = gaussianRand(posOnTexture + vec2(uTime, 1234.5678)) * 2.0 - 1.0;
    velRandom.z = gaussianRand(posOnTexture + vec2(uTime, 2468.9135)) * 2.0 - 1.0;

	// SCALE data to resolution
	int posScaledX = int(posRandom.x * (uRes.x-1));
	int posScaledY = int(posRandom.y * (uRes.x-1));
	int posScaledZ = int(posRandom.z * (uRes.y-1));
	vec3 posscaledtoresolution = ivec3(posScaledX, posScaledY, posScaledZ);

	vec4 dna = randomVec4Range(posOnTexture, uTime, 0.0, 1.0);
	
	Agent a;
	a.position = posscaledtoresolution;
	a.speed = velRandom;

	vec3 random_direction = normalize(vec3(random2(uTime), random2(uTime + 1234.5678), random2(uTime + 2468.9135)));
	a.direction = random_direction;

	a.acceleration = vec3(0);
	a.DNA.r = dna.r;
	a.DNA.g = dna.g;
	a.DNA.b = dna.b;
	a.DNA.a = dna.a; // ESPECE

	a.DNAScaled.r = Map(a.DNA.r, 0, 1, 0.1, 2); // RADIANS
	a.DNAScaled.g = Map(a.DNA.g, 0, 1,  -1, 1); // magnitude of the force
	a.DNAScaled.b = Map(a.DNA.b, 0, 1, 0.1, 3); 

	a.health = 1;

	a.maxSpeed = a.DNA.r * 2 + uSteerConfigs.y; // Adding 0.5 to make sure maxSpeed is always greater than 0.5
	a.minSpeed = a.DNA.r + uSteerConfigs.y; // Reducing the multiplication factor to make sure minSpeed is always less than maxSpeed

	a.mass = a.DNA.g * 3;
	a.size = a.DNA.b * 3;

    a.myCoord = posOnTexture;
	a.crouded = 0;

	if(a.DNA.a < 0.3)
	{
		a.espece = 1;
		a.color = vec4(1.0, 0, 0, a.health);
	}
	else if (a.DNA.a < 0.7)
	{
		a.espece = 2;
		a.color = vec4(0.7, 0, 1.0, a.health);
	}
	else if (a.DNA.a <= 1)
	{
		a.espece = 3;

		float colorRandom = texelFetch(sTD2DInputs[IN_INIT_POS], posOnTexture, 0).a;

		if(colorRandom < 0.4)
		{
			a.color = vec4(0.4, 1, 0.2, a.health);
		}
		else if(colorRandom < 0.6)
		{
			a.color = vec4(0.4, 1, 0.5, a.health);
		}
		else 
		{
			a.color = vec4(0.4, 1, 0.8, a.health);
		}
	}

	return a;
}

// REPRODUCTION AGENTS

Agent Reproduction(ivec2 posOnTexture, Agent a1, Agent a2)
{
	Agent child;
	
	float mutationRate = uMutationRate;

	child = ResetAgents(posOnTexture);
	float midpointRandom = random(posOnTexture.xx + sin(uTime * 2.49));
	float midpoint = floor(midpointRandom * 4);

	for(int i = 0; i < 4; i++)
	{
		if(i > midpoint)
		{
			child.DNA[i] = a1.DNA[i];
		}
		else 
		{
			child.DNA[i] = a2.DNA[i];
		}

		float mutationChance = random(child.position.xy + sin(uTime * .921));

		if((mutationChance < mutationRate))
		{
			child.DNA[i] = random(posOnTexture * cos(uTime * .1129));
		}
	}

	return child;
}

// ***** BEHAVIOR FUNCTIONS ***** //

// GENERAL
void ApplyForce(inout Agent a, vec3 force)
{
	a.acceleration += force / a.mass ;
}

void ApplyForce2D(inout Agent a, vec2 force) // inout is like a pointer
{
	a.acceleration.xy += force / a.mass;
}

void SeekTarget(inout Agent a, vec3 target, float mag, float rad)
{
	vec3 desired = target - a.position; // direction
	float dist = length(desired); 
	desired = normalize(desired); // use only the direction of the vector
	float newMaxSpeed = a.maxSpeed + uSteerConfigs.y;

	if(dist < rad/2) // very close to the target
	{
		desired *= Map(dist, 0, rad/2, 0, newMaxSpeed);
		vec3 steer = desired - a.speed;
		steer += 0.01 * randomVec3Range(a.myCoord, uTime, -1.0, 1.0);
		steer = clamp(steer, vec3(-newMaxSpeed), vec3(newMaxSpeed));
		ApplyForce(a, steer * mag);
	}
	else if(dist < rad) // close to the target
	{
		desired *= Map(dist, 0, rad, 0, newMaxSpeed);
		vec3 steer = desired - a.speed;
		steer += 0.01 * randomVec3Range(a.myCoord, uTime, -1.0, 1.0);
		steer = clamp(steer, vec3(-newMaxSpeed), vec3(newMaxSpeed));
		ApplyForce(a, steer * mag);
	}
	else if(dist < rad * 2 && dist >= rad) // within double the target radius
	{
		desired *= newMaxSpeed;
		vec3 steer = desired - a.speed;
		steer += 0.01 * randomVec3Range(a.myCoord, uTime, -1.0, 1.0);
		steer = clamp(steer, vec3(-newMaxSpeed), vec3(newMaxSpeed));
		ApplyForce(a, steer * mag);
	}
	else
	{
		return;
	}
}

void Attractors(inout Agent a)
{
	Agent b = a;

	for(int i = 0; i < uNumAttractors; i ++)
	{
		vec4 attractor = texelFetch(sTD2DInputs[IN_ATTRACTORS_BUFFER], ivec2(i, 0), 0);
		vec3 attractorPos = attractor.rgb;
		
		float dist = distance(attractorPos.xy, b.position.xy);

		if (dist < uSeekSettings.x - 4)
		{
			a.health -= 0.5;
		}
		else
		{
			// Use species-dependent behavior
			switch(a.espece) 
			{
				case 1: // species 1
					SeekTarget(a, attractor.xyz, a.DNAScaled.g * uAttractionConfigs * 1.2, a.DNAScaled.r * uSeekSettings.x);
					break;
				case 2: // species 2
					SeekTarget(a, attractor.xyz,  a.DNAScaled.g * uAttractionConfigs * 0.8, a.DNAScaled.r);
					break;
				case 3: // species 3
					SeekTarget(a, attractor.xyz,  a.DNAScaled.g * uAttractionConfigs, a.DNAScaled.r);
					break;
			}
		}
	}
}

vec3 Eat(Agent a, Food foods[7])
{
	float record = 99999999;
	int closest = -1;
	float d = 99999999;

	for(int i = 0; i < uCounts.y; i++)
	{		
		d = distance(a.position.xy, foods[i].position.xy);

		// DETECTER LE PLUS PROCHE
		if (d < record)
		{
			record = d;
			closest = i;
		}
	}

	return vec3(closest, record, d);
}

void RememberFood(inout Agent a, vec3 foodPos)
{
	a.foodMemory[a.memoryIndex] = foodPos;
	a.memoryIndex = (a.memoryIndex + 1) % 5; // Wrap around the memory index
}

void ApplyBoundaryForce(inout Agent agent)
{
    vec3 center = vec3(uRes.x / 2.0, uRes.x / 2.0, uRes.y / 2.0); // Assuming uRes is the size of the simulation space
    float boundary = 0.1 * uRes.x; // Boundary within 10% of the edge
    float dist = length(center - agent.position);

    if (dist > uRes.x - boundary)
    {
        vec3 force = normalize(center - agent.position);
        ApplyForce(agent, force);
    }

	// vec3 center = vec3(uRes.x / 2.0, uRes.x / 2.0, uRes.y / 2.0); 
    // float boundary = 0.1 * min(min(uRes.x, uRes.x), uRes.y); 
    // float dist = distance(center, agent.position);

    // if (dist > boundary)
    // {
    //     vec3 force = normalize(agent.position - center);
    //     ApplyForce(agent, force);
    // }
}

// Primordial Particles

vec3 calculateForce(inout Agent a, vec3 targetPos, float targetMass, float targetSpecies, float targetColor, float mapFactor) 
{

    vec3 force = targetPos - a.position; // direction
    float dist = distance(targetPos, a.position); // magnitude
    force = normalize(force); // use only the direction of the vector

    float distSq = dist*dist;
	float strength = (uG * mapFactor) * (a.mass * targetMass) / distSq;

    if(targetSpecies == 0.7 || targetSpecies == 1.0) 
    { 
        // ANTS or BOIDS
        //strength = (uG * mapFactor) * (a.mass * targetMass) / distSq;
		strength *= 1.0;  // no changes
    } 
    else if(targetSpecies == 0.4 && targetColor == 0.2 && dist > uMinMaxDistandCroudMin.x && dist < uMinMaxDistandCroudMin.y) 
    { 
        // PP REPULSION 0.2
        strength *= -2.5; // increased repulsion strength
    } 
    else if(targetSpecies == 0.4 && targetColor == 0.5 && dist > uMinMaxDistandCroudMin.x && dist < uMinMaxDistandCroudMin.y) 
    { 
        // PP 
        strength *= 0.5;
    } 
    else if(targetSpecies == 0.4) 
    {
        strength *= -1.5; // increased repulsion strength
    } 
    else 
    {
        return vec3(0, 0, 0); // No force for other cases
    }
    
    force = force * strength;
    vec3 steer = clamp(force, vec3(-uSteerConfigs.z), vec3(uSteerConfigs.z));

    return steer;
}

void GravitationalAttraction(inout Agent a)
{	
	Agent b = a;

	float mapLala = -2; // ANTS
	float mapLele = -3; // BOIDS
	float mapLulu = -1; // SELF

	if(uDNA == 1)
	{
		float lala  = b.DNA.r;
		mapLala = Map(lala, 0, 1, -5, 5);
		float lele  = b.DNA.g;
		mapLele = Map(lele, 0, 1, -5, 5);
		float lulu  = b.DNA.b;
		mapLulu = Map(lulu, 0, 1, -5, 5);
	}

    // goes through each neighborhood cell in range
	for (int x = -uRangeNeighbors.x; x <= uRangeNeighbors.x; x++)
	{
		for(int y = -uRangeNeighbors.x; y <= uRangeNeighbors.x; y++)
		{
			for(int z = -uRangeNeighbors.x; z <= uRangeNeighbors.x; z++)
			{
				if (!(x == 0 && y == 0 && z == 0)) // NOT CURRENT CELL
				{
                    vec3 targetPos = vec3(b.position.x + x, b.position.y + y, b.position.z + z);
					vec3 targetSpeciesAndColor = texelFetch(sTD2DInputs[IN_COLOR_AGENTS_BUFFER], ivec2(targetPos), 0).rba;
					float targetHealth = targetSpeciesAndColor.z;
					float targetSpecies = targetSpeciesAndColor.x;
					float targetColor = targetSpeciesAndColor.y;

                    if(targetHealth > 0) 
                    {
                        vec4 targetMassCoord = texelFetch(sTD2DInputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(targetPos), 0);
                        float targetMass = targetMassCoord.r;
                        
                        float mapFactor;
                        if(targetSpecies == 0.7) mapFactor = mapLala; // ANTS
                        else if(targetSpecies == 1.0) mapFactor = mapLele; // BOIDS
                        else if(targetSpecies == 0.4) mapFactor = mapLulu; // PP
                        else continue;
                        
                        vec3 steer = calculateForce(a, targetPos, targetMass, targetSpecies, targetColor, mapFactor);
                        ApplyForce(a, steer);

                        if (targetSpecies == 0.4 && distance(targetPos, b.position) <= uMinMaxDistandCroudMin.x)
					    {
						    a.crouded ++;
					    }
                    }
					else a.crouded --;
                }
            }
        }            
    }
}

// ANTS

void ScapeCroudDirection (inout Agent a, float mag) 
{
	float uThreshould = mag;

	for(int i = 0; i < uRangeCroud; i ++)
	{
		for(int j = 0; j < uRangeCroud; j++)
		{
			ivec2 uv = ivec2(i,j);

			Agent other = ReadAgent(uv);

			vec2 dir = other.position.xy - a.position.xy;
			float dist = length(dir);

			if(dist < 0.0001 || dist * dist > uAvoidDistance * uAvoidDistance)
			{
				continue;
			}
			else
			{
				// inside the else block where you're currently applying force
				float pct = dist * dist / uAvoidDistance * uAvoidDistance;
				pct = uThreshould / pct - 1.0;
				vec2 f = normalize(dir)* pct;
				vec3 newDirection = vec3(f, 0.0); // assuming z is not considered
				a.direction = mix(a.direction, newDirection, 0.1); // you can adjust this factor

				if(other.espece == a.espece)
				{
					// Apply crowd avoidance force and penalty for overcrowding
					ApplyForce2D(a, f);
					a.health += 0.001;
				}
				else
				{
					// Handle interspecies interaction: e.g., predator-prey relationship
					if(a.espece == 3 && other.espece == 1)
					{
						// Predator gains health, prey loses health
						a.health -= 0.001;
						other.health -= 0.001;
					}
                    else if(a.espece == 3 && other.espece == 2)
					{
						// Predator gains health, prey loses health
						a.health -= 0.001;
						other.health -= 0.001;
					}
				}
			}
		}
	}
}

vec2 AvoidBoids(Agent a)
{
    vec2 avoidanceForce = vec2(0);
	float uAvoidBoidDistance = 2;

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            ivec2 uv = ivec2(i, j);

            Agent other = ReadAgent(uv);

            if (other.espece != a.espece)
            {
                vec2 dir = other.position.xy - a.position.xy;
                float dist = length(dir);

                if (dist < uAvoidBoidDistance)
                {
                    vec2 f = normalize(dir) * (1.0 - dist / uAvoidBoidDistance);
                    avoidanceForce += f;
                }
            }
        }
    }

    return avoidanceForce;
}

vec3 NeighborhoodTurns(Agent a)
{
    vec3 vectors[100];
    float maxTrail = uConfigsAnts.y;
    int i = 0;
	float uBoidAvoidanceFactor = 1.0;

    // goes through each neighborhood cell in range
    for (int x = -uRangeNeighbors.z; x <= uRangeNeighbors.z; x++)
    {
        for (int y = -uRangeNeighbors.z; y <= uRangeNeighbors.z; y++)
        {
            if (!(x == 0 && y == 0)) // NOT CURRENT CELL
            {
                vec2 direction = vec2(x, y);

                if (dot(normalize(direction), a.direction.xy) > uConfigsAnts.x) // sensing angle
                {
                    ivec2 coord = ivec2(round(a.position.xy + direction));

                    // samples the trail level at that coordinate
                    vec3 level = texelFetch(sTD2DInputs[IN_TRAILS_BUFFER_IN], ivec2(coord), 0).rgb;

                    if (level.r == maxTrail && level.g == maxTrail && level.b == maxTrail)
                    {
                        // adds the directions to the vector's list
                        vectors[i] = normalize(vec3(x, y, 0));
                        i++;
                    }
                    else if (level.r > maxTrail && level.g > maxTrail && level.b > maxTrail)
                    {
                        maxTrail = level.r;
                        i = 0;
                        vectors[i] = normalize(vec3(x, y, 0));
                        i++;
                    }
                }
            }
        }
    }

    vec3 previousDirection = a.direction;
    vec3 currentDirection = normalize(a.speed.xyz);
    float mixFactor = 0.1; // Adjust this factor to control the weight of the previous direction.

    vec3 d = mix(currentDirection, previousDirection, mixFactor);

    if (maxTrail >= 0.01)
    {
        float randi = random2(sin(uTime * 9.2));
        float randIndex = Map(randi, 0, 1, 0, i - 1);
        int index = (i - 1) - int(randIndex);
        d = d + vectors[index] * 0.9;

        // Avoid areas with boids
        vec3 boidAvoidanceForce = vec3(AvoidBoids(a), 1.0);
        d += boidAvoidanceForce * uBoidAvoidanceFactor;
    }
    else
    {
        d = vec3(random2(sin(uTime * 0.2)), random2(cos(uTime * 5.2)), random2(sin(uTime * 14.2)));
        d = map_vector(d, -1, 1);
    }

    d = normalize(d);

    return d * 4;
}

// BOIDS

vec3 BoidBehavior (Agent a, int boidRule)
{
	Agent b = a;
	float angle = uSeekSettings.y;

	int range = uRangeNeighbors.y; // sensing distance 
	int total = 0;
	vec3 steer = vec3(0);

	vec3 mydirection = normx(b.speed);
	vec3 direction, coord, neighborVelocity;
	float neighborEspece;
	// goes thorugh each neighborhood cell in range
	for (int x = -range; x <= range; x++)
	{
		for(int y = -range; y <= range; y++)
		{
			for(int z = -range; z <= range; z++)
			{
				if (!(x == 0 && y == 0 && z == 0)) // NOT CURRENT CELL
				{
					direction = vec3(x, y, z);
					

					if(dot(normx(direction), mydirection) > angle) // sensing angle
					{
						coord = b.position + direction;
						neighborVelocity = texelFetch(sTD2DInputs[IN_DRAW_SPEED_HEALTH_BUFFER], ivec2(coord), 0).rgb;
						neighborEspece = texelFetch(sTD2DInputs[IN_COLOR_AGENTS_BUFFER], ivec2(coord), 0).r;
						
						if(neighborEspece == 1)
						{
							if(boidRule == 1) // Align
							{
								if(neighborVelocity.r != 0 && neighborVelocity.g != 0 && neighborVelocity.b != 0)
								{
									steer += neighborVelocity;
									total ++;
								}
							}
							else if(boidRule == 2) // Cohesion
							{
								if(neighborVelocity.r != 0)
								{
									steer += coord;
									total ++;
								}
							}
							else if(boidRule == 3) // Separation
							{
								if(neighborVelocity.r > 0)
								{
									float dist = distance(b.position, coord);
									vec3 difference = b.position - coord;
									difference = difference / (dist * dist);

									steer += difference;
									total ++;
								}
							}
						}
					}
				}
			}	
		}
	}

	if (total > 0)
	{
		steer /= total; // number of neighbors verified
		steer = normx(steer);
		steer *= uSteerConfigs.y;
		steer -= b.speed;
	}

	return steer;
}

vec3 Align (Agent a)
{
	return BoidBehavior(a, 1);
}

vec3 Cohesion (Agent a)
{
	return BoidBehavior(a, 2);
}

vec3 Separation (Agent a)
{
	return BoidBehavior(a, 3);
}

// Other parts of the code remain unchanged


void UpdateAgent(inout Agent agent) 
{
    // Update speed and direction
    agent.speed += agent.acceleration;

	// Add some random noise to agent's speed
	vec3 noise = vec3(random2(agent.position.x), random2(agent.position.y), random2(agent.position.z)); // Assume random2() returns a random number between -1 and 1.
	agent.speed += noise * 0.1; // Tune the 0.1 factor to increase or decrease the noise intensity

    float speed = length(agent.speed);
    vec3 direction = normalize(agent.speed);

    speed = clamp(speed, agent.minSpeed, agent.maxSpeed);

    agent.speed = direction * speed;
    agent.direction = direction;
    agent.position += agent.speed;

    // Boundary wrapping
    agent.position.x = mod(agent.position.x, uRes.x);
    agent.position.y = mod(agent.position.y, uRes.x);
    agent.position.z = mod(agent.position.z, uRes.y);

    // Add a damping factor to the acceleration instead of setting it to zero.
    agent.acceleration *= 0.9; // You may adjust this damping factor as needed.
}

// WRITE DATA INTO TEXTURES

void Write (Agent a, Food foods[7], ivec2 coord, int closest)
{
	vec4 outDrawMassCoord = vec4(0);
	vec4 outDrawSpeedHealth = vec4(0);

	// #define IN_INIT_POS 0
	// #define IN_POSITION_AND_MASS_BUFFER 1
	// #define IN_VELOCITY_AND_SIZE_BUFFER 2
	// #define IN_DIRECTION_AND_HEALTH_BUFFER 3
	// #define IN_DNA_BUFFER 4
	// #define IN_COLOR_AGENTS_BUFFER 5
	// #define IN_STATE_BUFFER 6
	// #define IN_DRAW_FOOD_BUFFER 7
	// #define IN_DRAW_MASS_COORD_BUFFER 8
	// #define IN_DRAW_SPEED_HEALTH_BUFFER 9
	// #define IN_TRAILS_BUFFER_IN 10
	// #define IN_TRAILS_BUFFER_OUT 0
	// #define IN_ATTRACTORS_BUFFER 11

    // Check if agent's position is within the boundaries
    bool withinBoundaries = coord.x >= 0 && coord.x < uRes.x && coord.y >= 0 && coord.y < uRes.x;

    if (withinBoundaries) 
    {
        if(a.health <= 0)
        {
            vec4 zeroVec = vec4(0.0);
            imageStore(mTDComputeOutputs[IN_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(zeroVec));
            imageStore(mTDComputeOutputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(zeroVec));
            imageStore(mTDComputeOutputs[IN_COLOR_AGENTS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(zeroVec));
            imageStore(mTDComputeOutputs[IN_DIRECTION_AND_HEALTH_BUFFER], coord, TDOutputSwizzle(zeroVec));
            imageStore(mTDComputeOutputs[IN_DNA_BUFFER], coord, TDOutputSwizzle(zeroVec));
            imageStore(mTDComputeOutputs[IN_VELOCITY_AND_SIZE_BUFFER], coord, TDOutputSwizzle(zeroVec));
            imageStore(mTDComputeOutputs[IN_STATE_BUFFER], coord, TDOutputSwizzle(zeroVec));
            imageStore(mTDComputeOutputs[IN_POSITION_AND_MASS_BUFFER], coord, TDOutputSwizzle(zeroVec));

            imageStore(mTDComputeOutputs[IN_TRAILS_BUFFER_OUT], ivec2(round(a.position.xy)), TDOutputSwizzle(zeroVec));
        }

        else
        {
            imageStore(mTDComputeOutputs[IN_POSITION_AND_MASS_BUFFER], coord, TDOutputSwizzle(vec4(a.position, a.mass)));

            imageStore(mTDComputeOutputs[IN_VELOCITY_AND_SIZE_BUFFER], coord, TDOutputSwizzle(vec4(a.speed, a.size)));

            imageStore(mTDComputeOutputs[IN_DIRECTION_AND_HEALTH_BUFFER], coord, TDOutputSwizzle(vec4(a.direction, a.health)));

            imageStore(mTDComputeOutputs[IN_DNA_BUFFER], coord, TDOutputSwizzle(a.DNA));
            imageStore(mTDComputeOutputs[IN_COLOR_AGENTS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(a.color)));
            outDrawMassCoord = vec4(a.mass, a.myCoord, a.color.r);
            outDrawSpeedHealth = vec4(a.speed, a.health);
            imageStore(mTDComputeOutputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(outDrawMassCoord));
            imageStore(mTDComputeOutputs[IN_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(outDrawSpeedHealth));
            vec4 colorState = vec4(float(a.crouded), a.color.r, a.color.g, a.color.b);
            imageStore(mTDComputeOutputs[IN_STATE_BUFFER], coord, TDOutputSwizzle(colorState));
        }

    }
	
	// Check if food's position is within the boundaries
    bool foodWithinBoundaries = coord.x < uCounts.y && coord.y == 0;

    if (foodWithinBoundaries) 
    {
        if(coord.x < uCounts.y && coord.y == 0)
        {
            vec4 foodColor = vec4(foods[coord.x].foodColor.rgb, foods[coord.x].alive);
            imageStore(mTDComputeOutputs[IN_DRAW_FOOD_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
            imageStore(mTDComputeOutputs[IN_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
            imageStore(mTDComputeOutputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
        }
    }

}

void WriteDiffuseTexture(Agent a, ivec2 posOnBuffer)
{
	float uTrailReinforcementFactor = 0.01;

    // color of present pixel
    vec4 oc = texelFetch(sTD2DInputs[IN_COLOR_AGENTS_BUFFER], posOnBuffer, 0);
    vec4 colorPresentPixel = texelFetch(sTD2DInputs[IN_TRAILS_BUFFER_IN], posOnBuffer, 0);

    if (colorPresentPixel.r > 0.01 && colorPresentPixel.g > 0.01 && colorPresentPixel.b > 0.01 && colorPresentPixel.a > 0.01)
    {
        // Calculate the fading factor based on time
        colorPresentPixel *= uDecayFactorAnts;
        colorPresentPixel = clamp(colorPresentPixel, 0, 1);
    }

    float avg = 0;

    // look at surrounding 9 squares
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            // surrounding square coordinate
            ivec2 coord = (posOnBuffer + ivec2(x, y));

            // neighbor species
            float especeNeighbor = texelFetch(sTD2DInputs[IN_COLOR_AGENTS_BUFFER], coord, 0).r;

            // average it
            if (especeNeighbor == 0.7)
            {
                avg += texelFetch(sTD2DInputs[IN_TRAILS_BUFFER_IN], coord, 0).r;
            }
        }
    }

    // multiply for trail decay factor
    avg /= 9;

    // Reinforce existing trails
    avg += uTrailReinforcementFactor; // Increase the trail level by a small amount at each time step

    oc += vec4(avg * uDecayFactorAnts);
    oc = clamp(oc, 0, 1);

    // update
    imageStore(mTDComputeOutputs[IN_TRAILS_BUFFER_OUT], posOnBuffer, TDOutputSwizzle(colorPresentPixel + oc));
}

// MAIN FUNCTION

void main()
{
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

	// AGENT DATA
	Agent agent = ReadAgent(coord); // read present coord agent data
	Agent mate = GenerateEmptyAgent();

	// FOOD DATA
	Food foods[7];

	ReadFoods(foods); // read all foods for each coord
	int closest = -1;

	if (ActiveAgent()) // if agent exists
	{
		if (AliveAgent(agent))
		{
			if(uResetTextureAndAgents == 1)
			{
				agent = ResetAgents(coord);
			}

			vec3 foodTarget;
			float useRememberedProb = 0.8; // 50% chance to use remembered food location
			if (randomVec4Range(agent.myCoord, uTime, 0.0, 1.0).x < useRememberedProb && length(agent.rememberedFoodLocation) > 0) 
			{
				// Use remembered food location
				foodTarget = agent.rememberedFoodLocation;
			} 
			else 
			{
				// Find closest food
				vec3 closestAndRecord = Eat(agent, foods);
				int closest = int(round(closestAndRecord.x));
				if (closest != -1) {
					foodTarget = foods[closest].position;
				}
			}

			if (length(foodTarget) > 0) 
			{
				// Seek food target
				SeekTarget(agent, foodTarget, uSteerConfigs.z, uSeekSettings.z);
			}

			float DNA0scaled = 1;
			float DNA1scaled = 1;
			float DNA2scaled = 1;

			if(agent.espece == 1 ) // BOIDS
			{
				if(uDNA == 1)
				{
					DNA0scaled = Map(agent.DNA.g, 0, 1, 0.1, 3);
					DNA1scaled = Map(agent.DNA.g, 0, 1, 0.1, 3);
					DNA2scaled = Map(agent.DNA.b, 0, 1, 0.1, 3);
				}
				else
				{
					DNA0scaled = uFlockConfigs.x;
					DNA1scaled = uFlockConfigs.y;
					DNA2scaled = uFlockConfigs.z;
				}

				vec3 alignForce = Align(agent) * DNA0scaled * uFlockConfigs.x;
				vec3 cohesionForce = Cohesion(agent) * DNA1scaled * uFlockConfigs.y;
				vec3 separationForce = Separation(agent) * DNA2scaled * uFlockConfigs.z;

				ApplyForce(agent, alignForce + cohesionForce + separationForce);
				// ApplyForce(agent, cohesionForce);
				// ApplyForce(agent, separationForce);
				// ApplyForce(agent, alignForce);
			}

			else if(agent.espece == 2) // PHYSARUM
			{
				if(uDNA == 1)
				{
					DNA0scaled = Map(agent.DNA.r, 0, 1, 0.1, 3);
					DNA1scaled = Map(agent.DNA.g, 0, 1, 0.5, 20);
					DNA2scaled = Map(agent.DNA.b, 0, 1, 0, 2);

					if(DNA2scaled <= 0.1)
					{
						ScapeCroudDirection(agent, 0.8);
					}
				}

				vec3 trailForce = NeighborhoodTurns(agent) * DNA0scaled;
				ApplyForce(agent, trailForce);
			}

			else if (agent.espece == 3) // PP
			{
				if(uDNA == 1)
				{
					DNA0scaled = Map(agent.DNA.r, 0, 1, 0.001, 1);
					DNA1scaled = Map(agent.DNA.g, 0, 1, 0.001, 1);
					DNA2scaled = Map(agent.DNA.b, 0, 1, 0.001, 1);
				}

				// vec3 clampedDNA = clampVector(agent.DNA.rgb, 0.3);
				vec3 noise = uDelta * (DNA0scaled * uNoiseFactor.r) * curlNoise(agent.position * (DNA1scaled * uNoiseFactor.g) + uTime * (DNA2scaled * uNoiseFactor.b));

				ApplyForce(agent, noise * uNoiseFactor.a);
				
				GravitationalAttraction(agent);
				ScapeCroudDirection(agent, 0.9); // INTERESTING BEHAVIOR WHEN ACTIVATED : DESACTIVATED, like colision detection
			}
			
			if (closest > -1 && agent.health < 5)
			{
				SeekTarget(agent, foods[closest].position, uSteerConfigs.z, uSeekSettings.z);
			}

			if(uNumAttractors >= 1)
			{
				Attractors(agent);
			}
			
			
			ApplyBoundaryForce(agent);

			if(agent.health >= 5)
			{
				float healthRecord = 0;

				for (int x = -1; x <= 1; x++)
				{
					for(int y = -1; y <= 1; y++)
					{
						if (!(x == 0 && y == 0)) // NOT CURRENT CELL
						{
							vec2 neighborPos = vec2(coord.x + x, coord.y + y); 
							vec4 neighborInfo = texelFetch(sTD2DInputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(neighborPos), 0);
							ivec2 neighborCoord = ivec2(neighborInfo.gb);
							float neighborColor = neighborInfo.a;

							if(neighborColor > 0 && neighborColor == agent.color.r) //reproduction intra species
							{
								float neighborHealth = texelFetch(sTD2DInputs[IN_DIRECTION_AND_HEALTH_BUFFER], ivec2(neighborCoord), 0).r;

								if(neighborHealth > healthRecord)
								{
									healthRecord = neighborHealth;
									mate = ReadAgent(neighborCoord);
								}
							}
						}
					}
				}
			}

			UpdateAgent(agent);
		}
			
		else
		{
			float born = random2((sin(uTime* .849)));

			if(born < uReproductionChance)
			{
				Agent child = Reproduction(coord, agent, mate); // like reset agent
				agent = child;
			}
			else
			{
				agent = GenerateEmptyAgent();
			}

		}
	}

	else
	{
		agent = GenerateEmptyAgent();
	}

	WriteDiffuseTexture(agent, coord);
	Write(agent, foods, coord, closest);
	
}
