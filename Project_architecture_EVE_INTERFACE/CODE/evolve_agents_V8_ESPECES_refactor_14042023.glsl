// Example Compute Shader

uniform vec3 uRes;
uniform int uAgentCount;
uniform float uTime;
uniform vec3 uSeekSettings;
uniform int uResetTextureAndAgents;
uniform int uNumFood;
uniform float uAngle;
uniform ivec2 uNeighborhood;
uniform vec3 uFlockConfigs;
uniform float uDecayFactor;
uniform vec3 uConfigs;
uniform float uMaxTrailLimits;
uniform float iSeekTargetForce;

uniform samplerBuffer aAttractorTransRad;

uniform samplerBuffer sFoodPositions;

#define PI 3.14159265358979323846
#define HALFPI 1.57079632679489661923
#define TWOPI 6.283185307179586
#define deg2rad 0.01745329251

#define IN_POSITION_AND_MASS_BUFFER 0
#define IN_VELOCITY_BUFFER 1
#define IN_DIRECTION_BUFFER 2
#define IN_HEALTH_BUFFER 3
#define IN_DRAW_AGENTS_BUFFER 4
#define IN_DNA_BUFFER 5
#define IN_DRAW_FOOD_BUFFER 6
#define INDEX_TRAILS 7
#define INDEX_DIFFUSE 8


layout (local_size_x = 8, local_size_y = 8) in;

// CLASS VEHICLE
struct Agent
{
	vec3 position;
	vec3 direction;
	vec3 speed;
	float maxSpeed;
	vec3 acceleration;
	float mass;
	float size;
	float DNA[4];
	float health;
	int espece;
};

struct Food
{
	vec3 position;
	vec4 foodColor;
	float alive;
	int index;
};

// CLASS TARGET
struct Target
{
	vec3 position;
	float size;
};

// UTILITIE FUNCTIONS

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

float Map(float value, float inLo, float inHi, float outLo, float outHi)
{
	return outLo + (value - inLo) * (outHi - outLo) / (inHi - inLo);
}

float integralSmoothstep( float x, float T )
{
    if( x>T ) return x - T/2.0;
    return x*x*x*(1.0-x*0.5/T)/T/T;
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

// READING TEXTURES

Agent GenerateEmptyAgent()
{
	Agent agent;
	agent.position = vec3(-9999);
	agent.speed = vec3(0);
	agent.acceleration = vec3(0);
	agent.size = 0;
	agent.mass = 0;
	agent.maxSpeed = 0;
	agent.direction = vec3(0,0,0);
	agent.health = 0;
	agent.DNA[0] = 0;
	agent.DNA[1] = 0;
	agent.DNA[2] = 0;
	agent.DNA[3] = 0;

	return agent;
}

Agent Read (ivec2 coord)
{
	Agent a;
	a.acceleration = vec3(0);

	vec4 posAndMass = texelFetch(sTD2DInputs[IN_POSITION_AND_MASS_BUFFER], coord, 0);
	a.position = posAndMass.rgb;
	a.mass = posAndMass.a;

	vec4 speedAndMax = texelFetch(sTD2DInputs[IN_VELOCITY_BUFFER], coord, 0);
	a.speed = speedAndMax.rgb;
	a.maxSpeed = speedAndMax.a;

	vec4 direction = texelFetch(sTD2DInputs[IN_DIRECTION_BUFFER], coord, 0);
	a.direction = direction.rgb;

	vec4 dna = texelFetch(sTD2DInputs[IN_DNA_BUFFER], coord, 0);
	a.DNA[0] = dna.r;
	a.DNA[1] = dna.g;
	a.DNA[2] = dna.b;
	a.DNA[3] = dna.a;

	float healthIn = texelFetch(sTD2DInputs[IN_HEALTH_BUFFER], coord, 0).x;
	a.health = healthIn;

	if(a.DNA[3] < 0.5)
	{
		a.espece = 1;
	}
	else if (a.DNA[3] < 0.9)
	{
		a.espece = 2;
	}
	else
	{
		a.espece = 3;
	}

	return a;
}

void ReadFoods(inout Food foods[7])
{
	for(int i = 0; i < uNumFood; i++)
	{
		vec4 foodInfo = texelFetch(sFoodPositions, i);
		
		foods[i].position.xy =  abs(foodInfo.rg) * uRes.x;
		foods[i].position.z =  abs(foodInfo.b) * uRes.z;
		foods[i].foodColor =  foodInfo;
		foods[i].alive = foodInfo.a;
		foods[i].index = i;
	}
}

// RESET AGENTS

Agent ResetAgents(ivec2 posOnTexture)
{
	// READ agent data from random textures
	vec3 posfromrandomtex = vec3(random(posOnTexture + sin(uTime * .49)), random(posOnTexture + cos(uTime * .029)), random(posOnTexture + sin(uTime * .2279)));

	vec3 velfromrandom = 2 * vec3(random(posOnTexture + sin(uTime * .149)), random(posOnTexture + cos(uTime * .729)), random(posOnTexture + sin(uTime * .751))) - 1;

	float maxSpeedRandom = random((posOnTexture + sin(uTime * .01)));
	float massRandom = random((posOnTexture * .21 - sin(uTime* .02)));
	massRandom = Map (massRandom, 0, 1, 0.2, 3);
	maxSpeedRandom = Map (maxSpeedRandom, 0, 1, 0.02, 1);

	// SCALE data to resolution
	int posScaledX = int(posfromrandomtex.x * (uRes.x-1));
	int posScaledY = int(posfromrandomtex.y * (uRes.y-1));
	int posScaledZ = int(posfromrandomtex.z * (uRes.z-1));
	vec3 posscaledtoresolution = ivec3(posScaledX, posScaledY, posScaledZ);

	vec4 randomDir = vec4( random(gl_GlobalInvocationID.xy * uTime), random(gl_GlobalInvocationID.xx * sin(uTime)), random(gl_GlobalInvocationID.yy * uTime * 0.58), 1);

	vec4 dna = vec4(random(posOnTexture.xx + cos(uTime * .49)), random(posOnTexture.xy + sin(uTime * .899)), random(posOnTexture.yy * sin(uTime * .0979)), random(posOnTexture.yx + sin(uTime * 0.265)));
	
	Agent a;
	a.position = posscaledtoresolution;
	a.speed = velfromrandom;
	a.direction = randomDir.rgb;
	a.direction = vec3(0);
	a.maxSpeed = maxSpeedRandom;
	a.mass = massRandom;
	a.acceleration = vec3(0);
	a.DNA[0] = dna.r;
	a.DNA[1] = dna.g;
	a.DNA[2] = dna.b;
	a.DNA[3] = dna.a; // ESPECE
	a.health = 1;

	if(a.DNA[3] < 0.5)
	{
		a.espece = 1;
	}
	else if (a.DNA[3] < 0.9)
	{
		a.espece = 2;
	}
	else
	{
		a.espece = 3;
	}

	return a;
}

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

// BEHAVIOR FUNCTIONS

vec3 Seek(vec3 targetPos, Agent agent)
{
	float rad = uSeekSettings.x;
	float maxsteerforce = uSeekSettings.y;
	float mag = uSeekSettings.z;

	vec3 desiredPos = targetPos - agent.position;

	float dist = length(desiredPos);
	desiredPos = normalize(desiredPos);

	if (dist < rad)
	{
		float forceMagnitude = Map(dist, 0, rad, 0, agent.maxSpeed.x);
		desiredPos *= forceMagnitude;
	}
	else
	{
		desiredPos *= agent.maxSpeed.x;
	}

	vec3 steer = desiredPos - agent.speed;
	steer = clamp(steer, vec3(-maxsteerforce), vec3(maxsteerforce));

	steer *= mag;

	return steer;
}

void ApplyForce(inout Agent a, vec3 force)
{
	a.acceleration += force / a.mass ;
}

vec3 Arrive(vec3 targetPos, Agent agent)
{
	float rad = uSeekSettings.x;
	float maxsteerforce = uSeekSettings.y;
	float mag = uSeekSettings.z;

	vec3 desiredPos = targetPos - agent.position;

	float dist = length(desiredPos);
	desiredPos = normx(desiredPos);

	vec3 steer;

	if (dist < rad)
	{
		float m = Map(dist, 0, rad, 0, agent.maxSpeed.x);
		desiredPos = normx(desiredPos);
		desiredPos *= m;

		steer = desiredPos - agent.speed;
		steer = clamp(steer, vec3(-maxsteerforce), vec3(maxsteerforce));

		steer *= mag;
	}
	else
	{
		//steer = Align(agent);
	}

	return steer;

}

vec3 Align (Agent a)
{
	Agent b = a;
	float angle = uAngle;
	vec3 avg = vec3(0);

	int range = uNeighborhood.x; // sensing distance 
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
						vec4 neighborVelocity = texelFetch(sTD2DInputs[IN_DRAW_AGENTS_BUFFER], coord, 0);
						
						if(neighborVelocity.r != 0 && neighborVelocity.b != 0 && neighborVelocity.a != 0)
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
		avg *= b.maxSpeed;
		steer = avg - b.speed;
		steer = clampVector(steer, maxsteerforce);
	}

	return steer;
}

vec3 Cohesion (Agent a)
{
	Agent b = a;
	float angle = uAngle;
	vec3 avg = vec3(0);

	int range = uNeighborhood.x; // sensing distance 
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
						vec3 neighborVelocity = texelFetch(sTD2DInputs[IN_DRAW_AGENTS_BUFFER], ivec2(coord), 0).rgb;
						
						if(neighborVelocity.r != 0)
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
		steer *= b.maxSpeed;
		steer -= b.speed;
		steer = clampVector(steer, maxsteerforce);
	}

	return steer;
}

vec3 Separation (Agent a)
{
	Agent b = a;
	float angle = uAngle;
	vec3 avg = vec3(0);

	int range = uNeighborhood.y; // sensing distance 
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
						vec3 neighborVelocity = texelFetch(sTD2DInputs[IN_DRAW_AGENTS_BUFFER], ivec2(coord), 0).rgb;
						
						if(neighborVelocity.r > 0)
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
		//steer = avg - b.position;
		steer = normx(avg);
		steer *= b.maxSpeed;
		steer -= b.speed;
		steer = clampVector(steer, maxsteerforce);
	}

	return steer;
}

vec3 Eat(Agent a, Food foods[7])
{
	float record = 99999999;
	int closest = -1;
	float d = 99999999;
	float perceptionRadius = 5;

	for(int i = 0; i < uNumFood; i++)
	{
		// SI LA NOURRITURE N'EST PAS DEJA MANGEE
		if (foods[i].alive > 0)
		{
			d = distance(a.position.xy, foods[i].position.xy);

			// DETECTER LE PLUS PROCHE
			if (d < record)
			{
				record = d;
				closest = i;
			}
		}
	}

	return vec3(closest, record, d);
}

vec3 NeighborhoodTurns (Agent a)
{
	vec3 vectors[50];
	float maxTrail = uMaxTrailLimits;
	int range = int(uConfigs.y); // sensing distance 
	int i = 0;

	// goes thorugh each neighborhood cell in range
	for (int x = -range; x <= range; x++)
	{
		for(int y = -range; y <= range; y++)
		{
			if (!(x == 0 && y == 0)) // NOT CURRENT CELL
			{
				vec2 direction = vec2(x, y);

				if(dot(normalize(direction), a.direction.xy) > uConfigs.x) // sensing angle
				{

					ivec2 coord = ivec2(round(a.position.xy + direction));

					// samples the trail level at that coordinate
					//ivec2 lookUpInfoAt = ivec2(round(a.position + a.direction * 2));
					vec3 level = texelFetch(sTD2DInputs[INDEX_DIFFUSE], ivec2(coord), 0).rgb;

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

vec3 RandomWalk(vec3 targetPos, Agent a)
{
	float r = random((a.position.xy * .91 - sin(uTime* .02)));
	vec3 randomForce = vec3(0);

	if(r < 0.2)
	{
		randomForce = targetPos;
	}
	else if (r < 0.8)
	{
		randomForce.x = a.position.x + Map(random(a.position.xx + sin(uTime * .08911)), 0, 1, -1, 1);
		randomForce.y = a.position.y + Map(random(a.position.yy + cos(uTime * .011)), 0, 1, -1, 1);
	}
	else
	{
		randomForce.z = targetPos.z + (2 * random(a.position.xy + cos(uTime * .729)) - 1);
	}

	return randomForce;
}

void RandomWalkFood (inout Food foods[7])
{
	for(int i = 0; i < uNumFood; i++)
	{
		int stepx = int(Map(round(random(foods[i].position.xx + sin(uTime * .411))), 0, 1, -1, 1));
		int stepy = int(Map(round(random(foods[i].position.yy + sin(uTime * .368))), 0, 1, -1, 1));
		int stepz = int(Map(round(random(foods[i].position.zz + cos(uTime * .118))), 0, 1, -1, 1));
		foods[i].position.x += stepx * 10;
		foods[i].position.y += stepy * 10;
		foods[i].position.z += stepz * 10;
	}
}

// UPDATE AGENT DATA

void UpdateAgent(inout Agent a)
{
	a.speed += a.acceleration;
	a.speed = clampVector(a.speed, a.maxSpeed);
	a.direction = normalize(a.speed);
	a.position += a.speed;
	a.acceleration = vec3(0);
	a.health -= 0.006;
	
	//boundary wrap
	if(a.position.x < 0)
	{
		a.position.x = uRes.x - 1;
	}
	else if (a.position.x > uRes.x - 1)
	{
		a.position.x = 0;
	}
	if(a.position.y < 0)
	{
		a.position.y = uRes.y - 1;
	}
	else if (a.position.y > uRes.y - 1)
	{
		a.position.y = 0;
	}
	if(a.position.z < 0)
	{
		a.position.z = uRes.z - 1;
	}
	else if (a.position.z > uRes.z - 1)
	{
		a.position.z = 0;
	}
}

// WRITE DATA INTO TEXTURES

void Write (Agent a, Food foods[7], ivec2 coord, int closest)
{
	vec4 outPositionMass  = vec4(a.position, a.mass);
	imageStore(mTDComputeOutputs[IN_POSITION_AND_MASS_BUFFER], coord, TDOutputSwizzle(outPositionMass));

	vec4 outSpeedAndMax  = vec4(a.speed, a.maxSpeed);
	imageStore(mTDComputeOutputs[IN_VELOCITY_BUFFER], coord, TDOutputSwizzle(outSpeedAndMax));

	vec4 outDirection  = vec4(a.direction, 0);
	imageStore(mTDComputeOutputs[IN_DIRECTION_BUFFER], coord, TDOutputSwizzle(outDirection));

	vec4 outDraw = vec4(0);
	
	if(a.health <= 0)
	{
		imageStore(mTDComputeOutputs[IN_DRAW_AGENTS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
	}
	else
	{
		if( a.espece == 1)
		{
			outDraw  = vec4(a.speed.x, a.speed.y, a.speed.z, a.health);
		}
		else if (a.espece == 2)
		{
			outDraw  = vec4(0, 1, 0, a.health);
		}
		else
		{
			outDraw  = vec4(1, 0, 1, a.health);
		}

		imageStore(mTDComputeOutputs[IN_DRAW_AGENTS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(outDraw));
	}

	// vec4 outTarget  = vec4(0, 0, 1, 1);
	// imageStore(mTDComputeOutputs[IN_DRAW_AGENTS_BUFFER], ivec2(round(target.position.xy)), TDOutputSwizzle(outTarget));

	if(coord.x < uNumFood && coord.y == 0)
	{
		vec4 foodColor = vec4(foods[coord.x].foodColor.rgb, foods[coord.x].alive);

		// if(coord.x == closest)
		// {
		// 	imageStore(mTDComputeOutputs[IN_DRAW_FOOD_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(vec4(0)));
		// 	imageStore(mTDComputeOutputs[IN_FOOD_BUFFER], coord, TDOutputSwizzle(vec4(0)));
		// }
		// else
		// {
			imageStore(mTDComputeOutputs[IN_DRAW_FOOD_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
		//}
	}

	vec4 outDNA  = vec4(a.DNA[0], a.DNA[1], a.DNA[2], a.DNA[3]);
	imageStore(mTDComputeOutputs[IN_DNA_BUFFER], coord, TDOutputSwizzle(outDNA));

	vec4 outHealth  = vec4(a.health, 0, 0, 0);
	imageStore(mTDComputeOutputs[IN_HEALTH_BUFFER], coord, TDOutputSwizzle(outHealth));

	///////////////////////////

	// vec4 newTrailColor = vec4(0.1, 0.1, 0.1, 0.01);
	// vec4 readColorTrails = texelFetch(sTD2DInputs[INDEX_TRAILS], ivec2(gl_GlobalInvocationID.xy), 0);
	// vec4 readColorAgents = texelFetch(sTD2DInputs[IN_DRAW_AGENTS_BUFFER], ivec2(gl_GlobalInvocationID.xy), 0);

	// if (readColorAgents.g > 0.05  && trailColor.r >= 0.001)
	// {
	// 	trailColor.a += 0.001;
	// 	trailColor.rgb += 0.01;

	// 	imageStore(mTDComputeOutputs[INDEX_TRAILS], ivec2(gl_GlobalInvocationID.xy), TDOutputSwizzle(trailColor));
	// }
	// else if (readColorAgents.g > 0.05 && trailColor.r == 0)
	// {
	// 	imageStore(mTDComputeOutputs[INDEX_TRAILS], ivec2(gl_GlobalInvocationID.xy), TDOutputSwizzle(newTrailColor));
	// }
	// else 
	// {	
	// 	imageStore(mTDComputeOutputs[INDEX_TRAILS], ivec2(gl_GlobalInvocationID.xy), TDOutputSwizzle(trailColor));
	// }
}


void DiffuseTexture (ivec2 posOnBuffer, Target target)
{
	// color of present pixel
	vec4 oc = texelFetch(sTD2DInputs[IN_DRAW_AGENTS_BUFFER], posOnBuffer, 0);

	float avg = 0;

	// look at surrounding 9 squares
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			// surrounding square coordinate
			ivec2 coord = (posOnBuffer + ivec2(x, y));
			// avarage it
			avg += texelFetch(sTD2DInputs[INDEX_DIFFUSE], coord, 0).g;
		}
	}

	// multiply for trail decay factor
	avg /= 9;
	oc += vec4(avg * uDecayFactor);
	oc = clamp(oc, 0, 1);

	vec2 hitXY = target.position.xy;
	float brushSize = 10;

	if((hitXY.x != 0 && hitXY.y != 0) && distance(hitXY, posOnBuffer) < brushSize)
	{
		oc += 10;
	}

	// update
	imageStore(mTDComputeOutputs[INDEX_DIFFUSE], posOnBuffer, TDOutputSwizzle(oc));

}

// MAIN FUNCTION
void main()
{
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

	// AGENT DATA
	Agent agent = Read(coord); // read present coord agent data
	Agent mate = GenerateEmptyAgent();

	// FOOD DATA
	Food foods[7];
	ReadFoods(foods); // read all foods for each coord
	int closest = -1;

	// MOUSE TARGET DATA
	// Target target;
	// target.position = uTargetPos.rgb * uRes.x;
	// target.size = uTargetPos.a;

	if (Active()) // if agent exists
	{
		if (Alive(agent))
		{
			if(uResetTextureAndAgents == 1)
			{
				agent = ResetAgents(coord);
			}

			vec3 closestAndRecord = Eat(agent, foods); // evaluate clsoest food, if far, seek, if close, eat
			int closest = int(round(closestAndRecord.x));
			float record = closestAndRecord.y;
			float dist =  closestAndRecord.z;

			if (record < 1 || dist < 1)
			{
				//foods[closest].alive = -1;
				//foods[closest].foodColor = vec4(0);

				// imageStore(mTDComputeOutputs[IN_DRAW_FOOD_BUFFER], ivec2(foods[closest].position.xy), TDOutputSwizzle(foods[closest].foodColor));
				// imageStore(mTDComputeOutputs[IN_FOOD_BUFFER], ivec2(closest, 0), TDOutputSwizzle(foods[closest].foodColor));

				agent.health += 0.09;
			}

			vec3 seekFoodForce = vec3(0);
			vec3 seekTargetForce = vec3(0);

			//float ddd = distance(agent.position, target.position);
			float DNA0scaled = Map(agent.DNA[0], 0, 1, 1, 2);
			float DNA1scaled = Map(agent.DNA[1], 0, 1, 1, 2);
			float DNA2scaled = Map(agent.DNA[2], 0, 1, 1, 2);

			if(agent.espece == 1)
			{
				vec3 alignForce = Align(agent) * DNA0scaled;
				vec3 cohesionForce = Cohesion(agent) * DNA1scaled;
				vec3 separationForce = Separation(agent) * DNA2scaled;

				ApplyForce(agent, alignForce);
				ApplyForce(agent, cohesionForce);
				ApplyForce(agent, separationForce);
			}

			else if(agent.espece == 2)
			{
				vec3 trailForce = NeighborhoodTurns(agent) * agent.DNA[0];
				ApplyForce(agent, trailForce);
			}

			else if (agent.espece == 3)
			{
				vec3 randomPositionTarget = RandomWalk(agent.position, agent);
				vec3 seekRandomForce = Seek(randomPositionTarget, agent) * agent.DNA[0];
				ApplyForce(agent, seekRandomForce);
			}
			
			if (closest > -1 || agent.health <= 1)
			{
				seekFoodForce = Seek(foods[closest].position, agent) * agent.DNA[2];
				ApplyForce(agent, seekFoodForce);
			}

			if ((1 < agent.health && agent.health < 20)) // used to include DNA scaled
			{
				// seekTargetForce = Seek(target.position, agent) * agent.DNA[1];
				// ApplyForce(agent, seekTargetForce);
			}

			if(agent.health >= 10)
			{
				float healthRecord = 0;

				for (int x = -1; x <= 1; x++)
				{
					for(int y = -1; y <= 1; y++)
					{
						if (!(x == 0 && y == 0)) // NOT CURRENT CELL
						{
							ivec2 neighborCoord = ivec2(coord.x + x, coord.y + y); 
							vec4 colorNeighbor = texelFetch(sTD2DInputs[IN_DRAW_AGENTS_BUFFER], ivec2(neighborCoord), 0);

							if((colorNeighbor.r > 0 || colorNeighbor.g > 0 || colorNeighbor.b > 0 || colorNeighbor.a > 0))
							{
								float neighborHealth = texelFetch(sTD2DInputs[IN_HEALTH_BUFFER], ivec2(neighborCoord), 0).r;

								if(neighborHealth > healthRecord)
								{
									healthRecord = neighborHealth;
									mate = Read(neighborCoord);
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

			if(born >= 0.949)
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

	//RandomWalkFood(foods);

	//vec4 readColorTrails = texelFetch(sTD2DInputs[INDEX_TRAILS], ivec2(gl_GlobalInvocationID.xy), 0);
	//DiffuseTexture (coord, target);
	Write(agent, foods, coord, closest);
	
}
