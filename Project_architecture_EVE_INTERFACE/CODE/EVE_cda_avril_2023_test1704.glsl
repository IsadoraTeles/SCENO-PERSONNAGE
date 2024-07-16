
uniform float uTime;
uniform vec2 uRes;
uniform vec2 uCounts; 					// uAgentCount, uFoodCount
uniform int uResetTextureAndAgents;
uniform vec3 uSpeedConfigs; 			// uVelDamping, uMaxSpeed, uAbsMaxSteerForce
uniform vec2 uAttractionConfigs;		// uNumAttractors, uAttractionForce
uniform ivec3 uRangeNeighbors; 			// uRangeGravitation, uRangeFlock, uRangeAnts
uniform float uG;
uniform vec3 uMinMaxDistandCroudMin; 	// uMinDist, uMaxDist, uCroudMinDist
uniform vec3 uConfigsAnts; 				// uSensAngleAnts, // uSensDistAnts, uMaxTrailLimits
uniform vec3 uFlockConfigs;	 			// uAlign, uCohesion, uSeparation
uniform vec3 uSeekSettings; 			// uRad, uAngleFlock, // uSeekTargetForce
uniform float uDecayFactorAnts;
uniform vec4 uNoiseFactor;
uniform float uDelta;
uniform int uRangeEat;
uniform int uRangeCroud;
uniform vec2 utestGrav;


uniform samplerBuffer aAttractorTransRad;

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

layout (local_size_x = 8, local_size_y = 8) in;

// CLASS AGENT
struct Agent
{
	vec3 position;
	vec3 direction;
    vec4 color;
	vec3 speed;
	float maxSpeed;
	vec3 acceleration;
	float mass;
	float size;
	vec4 DNA;
	float health;
	int espece;
    ivec2 myCoord;
	int crouded;
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

float parabola( float x, float k )
{
    return pow( 4.0*x*(1.0-x), k );
}

float Map(float value, float inLo, float inHi, float outLo, float outHi)
{
	return outLo + (value - inLo) * (outHi - outLo) / (inHi - inLo);
}

float integralSmoothstep( float x, float T )
{
    if( x>T ) return x - T/2.0;
    return x*x*x*(1.0-x*0.5/T)/T/T;
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
	a.health = dirHealth.r;
	a.direction = dirHealth.gba;
	a.DNA = dna;
    a.myCoord = agentCoord;
	a.crouded = agentCrouded;

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
		a.color = vec4(0.4, 1, stateAndColor.a, a.health);
	}
	return a;
}


void ReadFoods(inout Food foods[7])
{
	for(int i = 0; i < uCounts.y; i++)
	{
		vec4 foodInfo = texelFetch(sFoodPositions, i);
		
		foods[i].position.xy =  foodInfo.rg;
		foods[i].position.z =  foodInfo.b;
		foods[i].foodColor =  foodInfo;
		//foods[i].alive = foodInfo.a;
		foods[i].index = i;
	}
}

Agent GenerateEmptyAgent()
{
	Agent agent;
	agent.position = vec3(-9999);
	agent.speed = vec3(0.0);
	agent.acceleration = vec3(0.0);
	agent.size = 0.0;
	agent.mass = 0.0;
	agent.maxSpeed = 0.0;
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
	// READ agent data from random textures
	vec3 posfromrandomtex = vec3(random(posOnTexture + sin(uTime * .49)), random(posOnTexture + cos(uTime * .029)), random(posOnTexture + sin(uTime * .2279)));

	vec3 velfromrandom = 2 * vec3(random(posOnTexture + sin(uTime * .149)), random(posOnTexture + cos(uTime * .729)), random(posOnTexture + sin(uTime * .751))) - 1;

	float maxSpeedRandom = random((posOnTexture + sin(uTime * .01)));
	float massRandom = random((posOnTexture * .21 - sin(uTime* .02)));
	massRandom = Map (massRandom, 0, 1, 0.5, 2);
	maxSpeedRandom = Map (maxSpeedRandom, 0, 1, 0.2, 1);

	// SCALE data to resolution
	int posScaledX = int(posfromrandomtex.x * (uRes.x-1));
	int posScaledY = int(posfromrandomtex.y * (uRes.x-1));
	int posScaledZ = int(posfromrandomtex.z * (uRes.y-1));
	vec3 posscaledtoresolution = ivec3(posScaledX, posScaledY, posScaledZ);

	vec4 randomDir = vec4( random(gl_GlobalInvocationID.xy * uTime), random(gl_GlobalInvocationID.xx * sin(uTime)), random(gl_GlobalInvocationID.yy * uTime * 0.58), 1);
	randomDir.r = Map (randomDir.r, 0, 1, -1, 1);
	randomDir.g = Map (randomDir.g, 0, 1, -1, 1);
	randomDir.b = Map (randomDir.b, 0, 1, -1, 1);
	randomDir.a = Map (randomDir.a, 0, 1, -1, 1);

	vec4 dna = vec4(random(posOnTexture.xx + cos(uTime * .49)), random(posOnTexture.xy + sin(uTime * .899)), random(posOnTexture.yy * sin(uTime * .0979)), random(posOnTexture.yx + sin(uTime * 0.265)));
	
	Agent a;
	a.position = posscaledtoresolution;
	a.speed = velfromrandom;
	a.direction = randomDir.rgb;
	a.direction = vec3(0);
	a.maxSpeed = maxSpeedRandom;
	a.mass = massRandom;
	a.acceleration = vec3(0);
	a.DNA.r = dna.r;
	a.DNA.g = dna.g;
	a.DNA.b = dna.b;
	a.DNA.a = dna.a; // ESPECE
	a.health = 1;
	a.size = 1;

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
	
	float mutationRate = 0.0001;

	child = ResetAgents(posOnTexture);
	float midpointRandom = random(posOnTexture.xx + sin(uTime * 2.49));
	float midpoint = midpointRandom * 4;

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

		if((mutationChance < mutationRate) || (child.DNA[i] == 0))
		{
			child.DNA[i] = random(posOnTexture * sin(uTime * .9129));
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
	vec3 desired = target - a.position;// direction
	float dist = length(desired); 
	desired = normalize(desired); // use only the direction of the vector

	if(dist < rad * 1.5)
	{
		float m = Map(dist, 0, rad, 0, uSpeedConfigs.y);
		desired *= m;
        vec3 steer = desired - a.speed;
		steer = clamp(steer, vec3(-uSpeedConfigs.z), vec3(uSpeedConfigs.z));
        ApplyForce(a, steer * mag);
	}
	else if(dist < rad * 4)
	{
		desired *= uSpeedConfigs.y;
		vec3 steer = desired - a.speed;
		steer = clamp(steer, vec3(-uSpeedConfigs.z), vec3(uSpeedConfigs.z));

		ApplyForce(a, steer * mag);
	}
	else
	{
		return;
	}
}

void Attractors(inout Agent a)
{
	for(int i = 0; i < uAttractionConfigs.x; i ++)
	{
		vec4 attractor = texelFetch(aAttractorTransRad, i);
		SeekTarget(a, attractor.xyz, uAttractionConfigs.y, uSeekSettings.x);
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

// Primordial Particles

void GravitationalAttraction(inout Agent a)
{
	
	Agent b = a;

    // goes thorugh each neighborhood cell in range
	for (int x = -uRangeNeighbors.x; x <= uRangeNeighbors.x; x++)
	{
		for(int y = -uRangeNeighbors.x; y <= uRangeNeighbors.x; y++)
		{
			for(int z = -uRangeNeighbors.x; z <= uRangeNeighbors.x; z++)
			{
				if (!(x == 0 && y == 0 && z == 0)) // NOT CURRENT CELL
				{
                    vec3 targetPos = vec3(b.position.x + x, b.position.y + y, b.position.y + z);
					vec3 targetSpeciesAndColor = texelFetch(sTD2DInputs[IN_COLOR_AGENTS_BUFFER], ivec2(targetPos), 0).rba;
					float targetHealth = targetSpeciesAndColor.z;
					float targetSpecies = targetSpeciesAndColor.x;
					float targetColor = targetSpeciesAndColor.y;

					if(targetHealth > 0 && targetSpecies > 0.7)
					{
						vec4 targetMassCoord = texelFetch(sTD2DInputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(targetPos), 0);

						float targetMass = targetMassCoord.r;
                        vec3 force = targetPos - b.position;// direction
                        float dist = distance(targetPos, b.position); // magnitude

						float distSq = dist*dist;
						force = normalize(force); // use only the direction of the vector

						float strenght = -uG * 1.8 * (a.mass * targetMass * 10) / distSq;

						force = force*strenght;

						vec3 steer = clamp(force, vec3(-uSpeedConfigs.z), vec3(uSpeedConfigs.z));

						ApplyForce(a, steer);
						
					}


                    if(targetHealth > 0 && targetSpecies == 0.4)
                    {
						
                   		vec4 targetMassCoord = texelFetch(sTD2DInputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(targetPos), 0);

						float targetMass = targetMassCoord.r;
                        vec3 force = targetPos - b.position;// direction
                        float dist = distance(targetPos, b.position); // magnitude

						float distSq = dist*dist;
						force = normalize(force); // use only the direction of the vector
						
						if(targetColor < 0.5 && dist > uMinMaxDistandCroudMin.x + 0.5 && dist < uMinMaxDistandCroudMin.y)
						{
							
							float strenght = uG * (a.mass * targetMass * 3) / distSq;

							force = force*strenght;

							vec3 steer = clamp(force, vec3(-uSpeedConfigs.z), vec3(uSpeedConfigs.z));

							ApplyForce(a, steer);
						}

                        else if(targetColor < 0.9 && dist > uMinMaxDistandCroudMin.x && dist < uMinMaxDistandCroudMin.y)
						{
							
							float strenght = (uG * 0.4) * (a.mass * targetMass * 4) / distSq;

							force = force*strenght;

							vec3 steer = clamp(force, vec3(-uSpeedConfigs.z), vec3(uSpeedConfigs.z));

							ApplyForce(a, steer);
						}

                        else 
						{
							
							float strenght = (-uG) * (a.mass * targetMass * 5) / distSq;

							force = force*strenght;

							vec3 steer = clamp(force, vec3(-uSpeedConfigs.z), vec3(uSpeedConfigs.z));

							ApplyForce(a, steer);
						}
						
						if (dist <= uMinMaxDistandCroudMin.z)
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

void ScapeCroudDirection (inout Agent a, float mag)
{
	Agent b = a;

    // goes thorugh each neighborhood cell in range
	for (int x = -uRangeCroud; x <= uRangeCroud; x++)
	{
		for(int y = -uRangeCroud; y <= uRangeCroud; y++)
		{
			for(int z = -uRangeCroud; z <= uRangeCroud; z++)
			{
				if (!(x == 0 && y == 0 && z == 0)) // NOT CURRENT CELL
				{
                    vec4 target = texelFetch(sTD2DInputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(b.position.x + x, b.position.y + y), 0);
                    
                    if(target.r == 0.0)
                    {
						vec2 targetPos = vec2(b.position.x + x, b.position.y + y);
                        vec2 direction = b.position.xy - targetPos;// direction
						direction = normalize(direction);
						ApplyForce2D(a, direction);
					}
				}
			}
		}
	}
}

// ANTS

vec3 NeighborhoodTurns (Agent a)
{
	vec3 vectors[50];
	float maxTrail = uConfigsAnts.z;
	int i = 0;

	// goes thorugh each neighborhood cell in range
	for (int x = -uRangeNeighbors.z; x <= uRangeNeighbors.z; x++)
	{
		for(int y = -uRangeNeighbors.z; y <= uRangeNeighbors.z; y++)
		{
			if (!(x == 0 && y == 0)) // NOT CURRENT CELL
			{
				vec2 direction = vec2(x, y);

				if(dot(normalize(direction), a.direction.xy) > uConfigsAnts.x) // sensing angle
				{

					ivec2 coord = ivec2(round(a.position.xy + direction));

					// samples the trail level at that coordinate
					//ivec2 lookUpInfoAt = ivec2(round(a.position + a.direction * 2));
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
	
	vec3 d = normalize(a.speed.xyz);
	if (maxTrail >= .1)
	{
		int index = (i - 1) * int(random((gl_GlobalInvocationID.xy * .21 - sin(uTime* .02))));
		d = d + vectors[index] * .9;
	}
	else
	{
		d = vec3(random(gl_GlobalInvocationID.xy * .91 - sin(uTime* .299)), random(gl_GlobalInvocationID.xy * .91 - cos(uTime* .122)), random(gl_GlobalInvocationID.xy * .91 - sin(uTime* .985)));
	}

	d = normalize(d);

	return d * 4;
}

// BOIDS

vec3 Align (Agent a)
{
	Agent b = a;
	float angle = uSeekSettings.y;
	vec3 avg = vec3(0);

	int range = uRangeNeighbors.y; // sensing distance 
	int total = 0;
	vec3 steer = vec3(0);

	float maxsteerforce = uSeekSettings.y;

	// goes thorugh each neighborhood cell in range
	for (int x = -range; x <= range; x++)
	{
		for(int y = -range; y <= range; y++)
		{
			for(int z = -range; z <= range; z++)
			{
				if (!(x == 0 && y == 0 && z == 0)) // NOT CURRENT CELL
				{
					vec3 direction = vec3(x, y, z);
					vec3 mydirection = normx(b.speed);

					if(dot(normx(direction), mydirection) > angle) // sensing angle
					{
						ivec2 coord = ivec2((b.position + direction));
						vec4 neighborVelocity = texelFetch(sTD2DInputs[IN_DRAW_SPEED_HEALTH_BUFFER], coord, 0);
						float neighborEspece = texelFetch(sTD2DInputs[IN_COLOR_AGENTS_BUFFER], coord, 0).r;
						
						if(neighborVelocity.r != 0 && neighborVelocity.b != 0 && neighborVelocity.a != 0 && neighborEspece == 1)
						{
							avg += neighborVelocity.rgb;
							total ++;
						}

					}
				}
			}	
		}
	}

	if (total > 0)
	{
		avg /= total; // number of neighbors verified
		avg = normx(avg);
		avg *= uSpeedConfigs.y;
		steer = avg - b.speed;
		//steer = clampVector(steer, maxsteerforce);
	}

	return steer;
}

vec3 Cohesion (Agent a)
{
	Agent b = a;
	float angle = uSeekSettings.y;
	vec3 avg = vec3(0);

	int range = uRangeNeighbors.y; // sensing distance 
	int total = 0;

	float maxsteerforce = uSeekSettings.y;
	vec3 steer = vec3(0);

	// goes thorugh each neighborhood cell in range
	for (int x = -range; x <= range; x++)
	{
		for(int y = -range; y <= range; y++)
		{
			for(int z = -range; z <= range; z++)
			{
				if (!(x == 0 && y == 0 && z == 0)) // NOT CURRENT CELL
				{
					vec3 direction = vec3(x, y, z);
					vec3 mydirection = normx(b.speed);

					if(dot(normx(direction), mydirection) > angle) // sensing angle
					{
						vec3 coord = vec3(b.position + direction);
						vec3 neighborVelocity = texelFetch(sTD2DInputs[IN_DRAW_SPEED_HEALTH_BUFFER], ivec2(coord), 0).rgb;
						float neighborEspece = texelFetch(sTD2DInputs[IN_COLOR_AGENTS_BUFFER], ivec2(coord), 0).r;

						
						if(neighborVelocity.r != 0 && neighborEspece == 1)
						{
							avg += coord;
							total ++;
						}
					}
				}
			}
		}
	}

	if (total != 0)
	{
		avg /= total; // number of neighbors verified
		steer = avg - b.position;
		steer = normx(steer);
		steer *= uSpeedConfigs.y;
		steer -= b.speed;
		//steer = clampVector(steer, maxsteerforce);
	}

	return steer;
}

vec3 Separation (Agent a)
{
	Agent b = a;
	float angle = uSeekSettings.y;
	vec3 avg = vec3(0);

	int range = uRangeNeighbors.y; // sensing distance 
	int total = 0;

	float maxsteerforce = uSeekSettings.y;
	vec3 steer = vec3(0);

	// goes thorugh each neighborhood cell in range
	for (int x = -range; x <= range; x++)
	{
		for(int y = -range; y <= range; y++)
		{
			for(int z = -range; z <= range; z++)
			{
				if (!(x == 0 && y == 0 && z == 0)) // NOT CURRENT CELL
				{
					vec3 direction = vec3(x, y, z);
					vec3 mydirection = normx(b.speed);

					if(dot(normx(direction), mydirection) > angle) // sensing angle
					{
						vec3 coord = vec3(b.position + direction);
						vec3 neighborVelocity = texelFetch(sTD2DInputs[IN_DRAW_SPEED_HEALTH_BUFFER], ivec2(coord), 0).rgb;
						float neighborEspece = texelFetch(sTD2DInputs[IN_COLOR_AGENTS_BUFFER], ivec2(coord), 0).r;
						
						if(neighborVelocity.r > 0 && neighborEspece == 1)
						{
							float dist = distance(b.position, coord);
							vec3 difference = b.position - coord;
							difference = difference / (dist * dist);

							avg += difference;
							total ++;
						}
					}
				}
			}
		}
	}

	if (total != 0)
	{
		avg /= total; // number of neighbors verified
		steer = avg - b.position;
		steer = normx(avg);
		steer *= uSpeedConfigs.y;
		steer -= b.speed;
		//steer = clampVector(steer, maxsteerforce);
	}

	return steer;
}

// UPDATE AGENT DATA

void UpdateAgent(inout Agent agent) // inout is like a pointer
{
	//boundary wrap
	if(agent.position.x < 0)
	{
		agent.position.x = uRes.x - 1;
	}
	else if (agent.position.x > uRes.x - 1)
	{
		agent.position.x = 0;
	}
	if(agent.position.y < 0)
	{
		agent.position.y = uRes.x - 1;
	}
	else if (agent.position.y > uRes.x - 1)
	{
		agent.position.y = 0;
	}
	if(agent.position.z < 0)
	{
		agent.position.z = uRes.y - 1;
	}
	else if (agent.position.z > uRes.y - 1)
	{
		agent.position.z = 0;
	}

	agent.health -= 0.001;
		
	float size = parabola(1.0 - agent.health, 1.0);

	agent.size =  Map(size, 0, 1, 0, 1);
	agent.speed += agent.acceleration;
	agent.speed *= uSpeedConfigs.x;
	agent.speed = clampVector(agent.speed, uSpeedConfigs.y);
	agent.direction = normalize(agent.speed);
	agent.position += agent.speed;
	agent.acceleration = vec3(0.0);

}

// WRITE DATA INTO TEXTURES

void Write (Agent a, Food foods[7], ivec2 coord, int closest)
{
	

	vec4 outDrawMassCoord = vec4(0);
	vec4 outDrawSpeedHealth = vec4(0);
	
	if(a.health <= 0)
	{
		imageStore(mTDComputeOutputs[IN_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[IN_COLOR_AGENTS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[IN_DIRECTION_AND_HEALTH_BUFFER], coord, TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[IN_DNA_BUFFER], coord, TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[IN_VELOCITY_AND_SIZE_BUFFER], coord, TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[IN_STATE_BUFFER], coord, TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[IN_POSITION_AND_MASS_BUFFER], coord, TDOutputSwizzle(vec4(0)));

	}

	else
	{
		imageStore(mTDComputeOutputs[IN_POSITION_AND_MASS_BUFFER], coord, TDOutputSwizzle(vec4(a.position, a.mass)));

		imageStore(mTDComputeOutputs[IN_VELOCITY_AND_SIZE_BUFFER], coord, TDOutputSwizzle(vec4(a.speed, a.size)));

		imageStore(mTDComputeOutputs[IN_DIRECTION_AND_HEALTH_BUFFER], coord, TDOutputSwizzle(vec4(a.health, a.direction)));

		imageStore(mTDComputeOutputs[IN_DNA_BUFFER], coord, TDOutputSwizzle(a.DNA));
		imageStore(mTDComputeOutputs[IN_COLOR_AGENTS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(a.color)));
		outDrawMassCoord = vec4(a.mass, a.myCoord, a.color.r);
		outDrawSpeedHealth = vec4(a.speed, a.health);
		imageStore(mTDComputeOutputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(outDrawMassCoord));
		imageStore(mTDComputeOutputs[IN_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(outDrawSpeedHealth));
		vec4 colorState = vec4(float(a.crouded), a.color.r, a.color.g, a.color.b);
		imageStore(mTDComputeOutputs[IN_STATE_BUFFER], coord, TDOutputSwizzle(colorState));
	}

	if(coord.x < uCounts.y && coord.y == 0)
	{
		vec4 foodColor = vec4(foods[coord.x].foodColor.rgb, foods[coord.x].alive);
		imageStore(mTDComputeOutputs[IN_DRAW_FOOD_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
		imageStore(mTDComputeOutputs[IN_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
		imageStore(mTDComputeOutputs[IN_DRAW_MASS_COORD_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
	}

}

void WriteDiffuseTexture (ivec2 posOnBuffer)
{
	// color of present pixel
	vec4 oc = texelFetch(sTD2DInputs[IN_COLOR_AGENTS_BUFFER], posOnBuffer, 0);

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

			// avarage it
			if(especeNeighbor == 0.7)
			{
				avg += texelFetch(sTD2DInputs[IN_TRAILS_BUFFER_IN], coord, 0).r ;
			}
		}
	}

	// multiply for trail decay factor
	avg /= 9;
	oc += vec4(avg * uDecayFactorAnts);
	oc = clamp(oc, 0, 1);

	// vec2 hitXY = target.position.xy;
	// float brushSize = 10;

	// if((hitXY.x != 0 && hitXY.y != 0) && distance(hitXY, posOnBuffer) < brushSize)
	// {
	// 	oc += 10;
	// }

	// update
	imageStore(mTDComputeOutputs[IN_TRAILS_BUFFER_OUT], posOnBuffer, TDOutputSwizzle(oc));

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

			vec3 closestAndRecord = Eat(agent, foods); // evaluate clsoest food, if far, seek, if close, eat
			int closest = int(round(closestAndRecord.x));
			float record = closestAndRecord.y;
			float dist =  closestAndRecord.z;

			if (record < 1 || dist < 2)
			{
				agent.health += 0.001;
			}

			vec3 seekFoodForce = vec3(0);
			vec3 seekTargetForce = vec3(0);

			//float ddd = distance(agent.position, target.position);
			// float DNA0scaled = Map(agent.DNA.r, 0, 1, 1, 2);
			// float DNA1scaled = Map(agent.DNA.g, 0, 1, 1, 2);
			// float DNA2scaled = Map(agent.DNA.b, 0, 1, 1, 2);

			if(agent.espece == 1 ) // BOIDS
			{
				vec3 alignForce = Align(agent) * uFlockConfigs.x;
				vec3 cohesionForce = Cohesion(agent) * uFlockConfigs.y;
				vec3 separationForce = Separation(agent) * uFlockConfigs.z;

				ApplyForce(agent, alignForce);
				ApplyForce(agent, cohesionForce);
				ApplyForce(agent, separationForce);

			}

			else if(agent.espece == 2) // PHYSARUM
			{
				vec3 trailForce = NeighborhoodTurns(agent);
				ApplyForce(agent, trailForce);
			}

			else if (agent.espece == 3) //NOT USING DNA YET
			{
				// vec3 noise = uDelta * uNoiseFactor.x * curlNoise(agent.position * uNoiseFactor.y + uTime * uNoiseFactor.z);

				// ApplyForce(agent, noise * uNoiseFactor.w);

				if(agent.crouded > uMinMaxDistandCroudMin.z)
				{
					ScapeCroudDirection(agent,  uAttractionConfigs.y);
				}
				if(agent.crouded < 0)
				{
					agent.crouded = 0;
				}

				GravitationalAttraction(agent);

			}
			
			if (closest > -1 && agent.health < 2)
			{
				SeekTarget(agent, foods[closest].position, uSpeedConfigs.y, uSeekSettings.x - 2);
			}
			
			Attractors(agent);

			if(agent.health >= 10)
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
			float born = random((coord * .21 - sin(uTime* .849)));

			if(born >= 0.989)
			{
				Agent child = Reproduction(coord, agent, mate); // like reset agent
				agent = child;
			}
		}
	}

	else
	{
		agent = GenerateEmptyAgent();
	}

	WriteDiffuseTexture(coord);
	Write(agent, foods, coord, closest);
	
}
