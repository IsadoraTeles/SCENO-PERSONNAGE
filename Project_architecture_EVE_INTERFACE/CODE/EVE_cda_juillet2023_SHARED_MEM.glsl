
uniform float uTime;
uniform vec2 uRes;                      // its a square with x and y bounds of 512px (uRes.x) and the z bound is two (uRes.y)
uniform vec2 uCounts; 					// uAgentCount, uFoodCount
uniform int uResetTextureAndAgents;
uniform vec3 uSteerConfigs; 			// uVelDamping, uMaxSpeed, uAbsMaxSteerForce
uniform int uAttractionConfigs;			// uAttractionForce
uniform ivec3 uRangeNeighbors; 			// uRangeGravitation, uRangeFlock, uRangeAnts
uniform float uG;
uniform vec2 uConfigsAnts; 				// uSensAngleAnts / DNA.g, // uMaxTrailLimits / DNA.r
uniform vec3 uFlockConfigs;	 			// uAlign / DNA.r, uCohesion / DNA.g, uSeparation / DNA.b
uniform vec3 uSeekSettings; 			// uRad, uAngleFlock, seekFoodForce
uniform float uDecayFactorAnts;			// DNA.b
uniform vec4 uNoiseFactor;				// DNA.r, DNA.g, DNA.b
uniform float uDelta;
uniform float uMutationRate;
uniform float uReproductionChance;
uniform int uNumAttractors;
uniform int uDNA;
uniform float uAvoidDistance;
uniform float uAvoidBoidsFactor;
uniform float uBalanceFactor;
uniform float uAlphaPPS;	//-100
uniform float uBetaPPS;		//-3
uniform int uRadiusPPS;		//10

#define INOUT_POS_MASS_BUFFER 0
#define INOUT_VEL_ESPECE_BUFFER 1

layout (local_size_x = 8, local_size_y = 8) in;

shared vec4 posMassShared[gl_WorkGroupSize.x][gl_WorkGroupSize.y];
shared vec4 velEspeceShared[gl_WorkGroupSize.x][gl_WorkGroupSize.y];

// CLASS AGENT
struct Agent
{
	// STORED VARIABLES
	vec3 position;
	vec3 direction;
    vec4 color;
	vec3 speed;
	float mass;
	float size;
	vec4 DNA;
	float health;
	int espece;
    ivec2 myCoord;
	vec3 foodMemory; // Memory of one food location
	float orientation; // ANGLE FOR EACH AGENT

	// NON STORED VARIABLES
	vec4 DNAScaled;
	float maxSpeed;
	float minSpeed;
	vec3 acceleration;
	//vec3 rememberedFoodLocation;
};

// UTILITIES FUNCTIONS

float random2(float seed)
{
    float value = fract(sin(seed) * 43758.5453);
    return value;
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

int Index (ivec2 coord)
{
	//ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 wh = ivec2(uTD2DInfos[0].res.zw); // uResolution
	int index = (coord.x + (coord.y * wh.x));
	return index;
}

bool ActiveAgent (ivec2 coord)
{
	return Index(coord) < uCounts.x;
}

bool AliveAgent(Agent a)
{
	return a.health > 0.0;
}

// READING TEXTURES

Agent ReadAgent(ivec2 coord)
{
	Agent a;
	vec4 dirHealth = texelFetch(sTD2DInputs[INOUT_DIRECTION_AND_HEALTH_BUFFER], coord, 0);
	a.health = dirHealth.a;
	
	vec4 posMass  = texelFetch(sTD2DInputs[INOUT_POSITION_AND_MASS_BUFFER], coord, 0);
	vec4 velSize = texelFetch(sTD2DInputs[INOUT_VELOCITY_AND_SIZE_BUFFER],coord, 0);
	vec4 dna = texelFetch(sTD2DInputs[INOUT_DNA_BUFFER], coord, 0);
	ivec2 agentCoord = coord;
	
	vec4 foodMemoryCOLORB_IN = texelFetch(sTD2DInputs[INOUT_MEMORYFOOD_COLOR_B_BUFFER], coord, 0);
	a.foodMemory = foodMemoryCOLORB_IN.xyz;

	// If foodMemory is the default value, set it to the center of the canvas
	if (a.foodMemory == vec3(-1, -1, -1)) 
	{
		a.foodMemory = vec3(uRes.x * 0.5, uRes.y * 0.5, 0.0);
	}

	float orientationIN = texelFetch(sTD2DInputs[IN_ORIENTATIONS_BUFFER], coord, 0).a;
	a.orientation = orientationIN;

	a.acceleration = vec3(0.0);
	a.position = posMass.rgb;
	a.mass = posMass.a;
	if(a.mass <= 0.1){a.mass = 0.2;}
	a.speed = velSize.rgb;
	a.size = velSize.a;
	a.direction = dirHealth.rgb;
	a.DNA = dna;
	a.myCoord = agentCoord;	

	a.maxSpeed = a.DNA.r * 2 + 0.5; // Adding 0.5 to make sure maxSpeed is always greater than 0.5
	a.minSpeed = a.DNA.r + 0.3; // Reducing the multiplication factor to make sure minSpeed is always less than maxSpeed

	if(a.DNA.a <= 0.3)
	{
		a.espece = 1;
		a.color = vec4(1.0, 0, 0, a.health);
	}
	else if (a.DNA.a <= 0.7)
	{
		a.espece = 2;
		a.color = vec4(0.7, 0, 1.0, a.health);
	}
	else if (a.DNA.a <= 1.0)
	{
		a.espece = 3;
		a.color.r = 0.4;
		a.color.g = texelFetch(sTD2DInputs[INOUT_COLOR_AGENTS_BUFFER], ivec2(a.position.xy), 0).g;
		a.color.a = a.health;

		if(a.DNA.a <= 0.8)
		{
			a.color.b = 0.1;
		}
		else if(a.DNA.a <= 0.9)
		{
			a.color.b = 0.5;
		}
		else 
		{
			a.color.b = 0.9;
		}
	}


	return a;
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
	agent.DNAScaled = vec4(3.0);
	agent.color = vec4(0.0);
	agent.espece = 0;
    agent.myCoord = ivec2(-9999);

	return agent;
}

// RESET AGENTS

Agent ResetAgents(ivec2 posOnTexture)
{
	vec3 posRandom = vec3(random2(posOnTexture.x * fract(uTime) * 0.05) * uRes.x, random2(posOnTexture.y * fract(uTime) * 0.001) * uRes.x, random2(posOnTexture.x * fract(uTime) * 0.0015) * uRes.y);

	//vec3 posRandom = texelFetch(sTD2DInputs[IN_INIT_POS], posOnTexture, 0).rgb;

	vec3 velRandom = vec3(random2(posOnTexture.x + 10 + fract(uTime)), random2(posOnTexture.y + 0.5 + fract(uTime) * 0.715), random2(posOnTexture.x * fract(uTime) * 0.915)) * 2.0 - 1.0;

	// SCALE data to resolution
	int posScaledX = int(posRandom.x);
	int posScaledY = int(posRandom.y);
	int posScaledZ = int(posRandom.z);
	vec3 posscaledtoresolution = vec3(posScaledX, posScaledY, posScaledZ);

	vec4 dna = vec4(random2(posOnTexture.x * fract(uTime) * 20.905), random2(posOnTexture.y * fract(uTime) * 9.001), random2(posOnTexture.x * fract(uTime) * 0.715), random2(posOnTexture.x * fract(uTime) * 9.905));
	//vec4 dna = texelFetch(sTD2DInputs[INOUT_DNA_BUFFER], posOnTexture, 0);

	Agent a;
	a.position = posscaledtoresolution;
	a.speed = velRandom;

	vec3 random_direction = vec3(random2(posOnTexture.x + uTime * 0.4), random2(posOnTexture.y + 0.7 + uTime * 0.915), random2(posOnTexture.x + uTime * 0.315));
	a.direction = normalize(random_direction) * 2.0 - 1.0;

	a.acceleration = vec3(0);
	a.DNA.r = dna.r;
	a.DNA.g = dna.g;
	a.DNA.b = dna.b;
	a.DNA.a = dna.a; // ESPECE

	a.DNAScaled.r = Map(a.DNA.r, 0, 1, uSeekSettings.x, 4 * uSeekSettings.x); // RADIANS
	a.DNAScaled.g = Map(a.DNA.g, 0, 1,  uAttractionConfigs, uAttractionConfigs * 5); // magnitude of the force
	a.DNAScaled.b = Map(a.DNA.b, 0, 1, 0.1, 3); 

	a.health = 2 * Map(random2(a.size + a.position.x * a.position.y * uTime * 0.191), 0.0, 1.0, 2.0,  100 * a.DNA.r);

	a.maxSpeed = a.DNA.r * 2.0 + uSteerConfigs.y; // Adding 0.5 to make sure maxSpeed is always greater than 0.5
	a.minSpeed = a.DNA.r + uSteerConfigs.y; // Reducing the multiplication factor to make sure minSpeed is always less than maxSpeed

	a.mass = a.DNA.g * 3;
	a.size = a.DNA.b * 3;

    a.myCoord = posOnTexture;

	// Initialize food memory to default
	a.foodMemory = vec3(-1, -1, -1);  

	if(a.DNA.a <= 0.3)
	{
		a.espece = 1;
		a.color = vec4(1.0, 0, 0, a.health);
		a.orientation = 0.0;
	}
	else if (a.DNA.a <= 0.7)
	{
		a.espece = 2;
		a.color = vec4(0.7, 0, 1.0, a.health);
		a.orientation = 0.0;
	}
	else if (a.DNA.a <= 1)
	{
		a.espece = 3;

		float colorRandom = random2(posOnTexture.x * uTime * 975);

		if(colorRandom < 0.4)
		{
			a.color = vec4(0.4, 1.0, 0.1, a.health);
			a.orientation = random2(Index(posOnTexture)* 0.01) * TWOPI; // random angles from 0 to 2PI and multiply by 3.14*2 to get full circle
		}
		else if(colorRandom < 0.6)
		{
			a.color = vec4(0.4, 1.0, 0.5, a.health);
			a.orientation = random2(Index(posOnTexture) * 0.01) * TWOPI;
		}
		else 
		{
			a.color = vec4(0.4, 1.0, 0.9, a.health);
			a.orientation = random2(Index(posOnTexture)* 0.01) * TWOPI;
		}
	}

	return a;
}

// ***** BEHAVIOR FUNCTIONS ***** //

// GENERAL
void ApplyForce(inout Agent a, vec3 force)
{
	a.acceleration += force / a.mass ;
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
		steer += 0.01 * randomVec3Range(a.myCoord, uTime * 0.0158, -1.0, 1.0);
		steer = clamp(steer, vec3(-newMaxSpeed), vec3(newMaxSpeed));
		ApplyForce(a, steer * mag);
	}
	else if(dist < rad) // close to the target
	{
		desired *= Map(dist, 0, rad, 0, newMaxSpeed);
		vec3 steer = desired - a.speed;
		steer += 0.01 * randomVec3Range(a.myCoord, uTime * 0.0858, -1.0, 1.0);
		steer = clamp(steer, vec3(-newMaxSpeed), vec3(newMaxSpeed));
		ApplyForce(a, steer * mag);
	}
	else if(dist < rad * 2 && dist >= rad) // within double the target radius
	{
		desired *= newMaxSpeed;
		vec3 steer = desired - a.speed;
		steer += 0.01 * randomVec3Range(a.myCoord, uTime * 0.948, -1.0, 1.0);
		steer = clamp(steer, vec3(-newMaxSpeed), vec3(newMaxSpeed));
		ApplyForce(a, steer * mag);
	}
	else
	{
		return;
	}
}

// PPS CODE

bool isOnRightSide(vec2 p, vec2 v, vec2 op) // position, speed and other position
{
    vec2 b = p + v;
    return ((b.x - p.x) * (op.y - p.y) - (b.y - p.y) * (op.x - p.x)) > 0.0;
}

vec2 updateLandR(Agent b)
{
	float left = 0.0;
	float right = 0.0;

	for (int x = -uRadiusPPS; x <= uRadiusPPS; x++)
    {
        for(int y = -uRadiusPPS; y <= uRadiusPPS; y++)
        {
			if (!(x == 0 && y == 0)) // NOT CURRENT CELL
			{
				vec2 targetPos = vec2(b.position.x + x, b.position.y + y);
				vec3 targetSpeciesAndColor = texelFetch(sTD2DInputs[INOUT_COLOR_AGENTS_BUFFER], ivec2(targetPos), 0).rba;
				float targetHealth = targetSpeciesAndColor.z;
				float targetSpecies = targetSpeciesAndColor.x;

				if(targetHealth > 0 && targetSpecies == 0.4)
				{
					// float targetColor = targetSpeciesAndColor.y;
					// ivec2 targetCoord = texelFetch(sTD2DInputs[INOUT_DRAW_MASS_COORD_BUFFER], ivec2(targetPos), 0).gb;

					if(isOnRightSide(b.position.xy, b.speed.xy, targetPos))
					{
						right += 1.0;
					}
					else
					{
						left += 1.0;
					}

				}
			}
			
		}
	}

	return vec2(right, left);
}

void ApplyOrientationForcePPSAgent(inout Agent agent, float magnitude)
{
	// Define a damping factor between 0 and 1
    // A factor of 0.1, for example, would reduce the velocity to 10% of the original
    float dampingFactor = 0.7;

	vec3 direction = vec3(cos(agent.orientation), sin(agent.orientation), 0.0);

	// PPS delta_phi algorithm
	float r = 0.0;
	float l = 0.0;

	vec2 rl = updateLandR(agent);
	r = rl.x;
	l = rl.y;

	float n = r + l;
	agent.color.g = Map(n, -30, 30, 0.0, 1.0);
	float delta_phi = radians(uAlphaPPS) + radians(uBetaPPS) * n * sign(r - l);

	// UPDATE agent orientation
	//agent.orientation += delta_phi;
	agent.orientation += delta_phi * dampingFactor;  // apply the damping factor


	direction = vec3(cos(agent.orientation), sin(agent.orientation), 0.0);

	agent.acceleration = direction * magnitude;  // You may need to define accelerationMagnitude, or replace it with an appropriate value.

}

// ANTS

vec3 AvoidBoids(Agent a)
{
    vec3 avoidanceForce = vec3(0);

    for (int x = -uRangeNeighbors.z - 1; x <= uRangeNeighbors.z + 1; x++)
    {
        for(int y = -uRangeNeighbors.z -1 ; y <= uRangeNeighbors.z + 1; y++)
        {
            for(int z = -uRangeNeighbors.z - 1; z <= uRangeNeighbors.z + 1; z++)
            {
                if (!(x == 0 && y == 0 && z == 0)) // NOT CURRENT CELL
                {

					vec3 targetPos = vec3(a.position.x + x, a.position.y + y, a.position.z + z);
                    vec3 targetSpeciesAndColor = texelFetch(sTD2DInputs[INOUT_COLOR_AGENTS_BUFFER], ivec2(targetPos), 0).rba;
                    
                    float targetSpecies = targetSpeciesAndColor.x;
                    float targetColor = targetSpeciesAndColor.y;
					float targetHealth = targetSpeciesAndColor.z;

					if (targetSpecies != a.color.r)
					{
						vec3 dir = targetPos - a.position;
						float dist = length(dir);

						if (dist < uAvoidDistance)
						{
							vec3 f = normalize(dir) * (1.0 - dist / uAvoidDistance);
							avoidanceForce += f;
						}
					}
				}
        	}
    	}
	}

    return avoidanceForce;
}

vec3 AntPhysarumBehavior(Agent a, float mixFactor, float forceFactor)
{
    // The avoidance force helps with Physarum like growth, i.e. avoidance of own species
    vec3 boidAvoidanceForce = vec3(AvoidBoids(a));

    vec3 maxTrailVector = vec3(0);
    vec3 vectors[100];
    float maxTrail = uConfigsAnts.y;
    int vectorCount = 0;

    // Sampling the neighborhood
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
                    vec3 level = texelFetch(sTD2DInputs[IN_TRAILS_BUFFER], ivec2(coord), 0).rgb;

                    if (level.r == maxTrail && level.g == maxTrail && level.b == maxTrail)
                    {
                        // adds the directions to the vector's list
                        vectors[vectorCount] = normalize(vec3(x, y, 0));
                        vectorCount++;
                    }
                    else if (level.r > maxTrail && level.g > maxTrail && level.b > maxTrail)
                    {
                        maxTrail = level.r;
                        vectorCount = 0;
                        vectors[vectorCount] = normalize(vec3(x, y, 0));
                        vectorCount++;
                    }
                }
            }
        }
    }

    vec3 previousDirection = a.direction;
    vec3 currentDirection = normalize(a.speed.xyz);
    vec3 d = mix(currentDirection, previousDirection, mixFactor); // Adjust this factor to control the weight of the previous direction.

    if (maxTrail >= 0.01)
    {
        // Select the direction from the vector list with max trail value (like ants)
        float randi = random2(sin(uTime * 9.2));
        float randIndex = Map(randi, 0, 1, 0, vectorCount - 1);
        int index = (vectorCount - 1) - int(randIndex);
        d = d + vectors[index] * 0.9;

        // Avoid areas with boids
        d += boidAvoidanceForce * uAvoidBoidsFactor;
    }
    else
    {
        // If there's no significant trail, behave like a physarum and explore randomly
        vec3 wanderDirection = vec3(random2(sin(uTime * 0.2)), random2(cos(uTime * 5.2)), random2(sin(uTime * 14.2)));
        wanderDirection = map_vector(wanderDirection, -1, 1);
        d = mix(d, wanderDirection, mixFactor); // Mix the current direction with a random direction
    }

    d = normalize(d);

    return d * forceFactor;
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

	vec3 mydirection = normx(b.speed);

	 for(int i = 0; i< gl_NumWorkGroups.x ; i++)
	 {
    	for (int j=0; j< gl_NumWorkGroups.y; j++)
		{
    	
    		ivec2 lookup = ivec2(gl_LocalInvocationID.x+gl_WorkGroupSize.x*i,gl_LocalInvocationID.y+gl_WorkGroupSize.y*j);
    		
    		posMassShared[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = texelFetch(sTD2DInputs[0], lookup, 0);
    	
    		memoryBarrier();
    		barrier();
    	
    		for(int k = 0; k< gl_WorkGroupSize.x ; k++)
			{
    			for (int l=0; l< gl_WorkGroupSize.y; l++)
				{
		    	
					//vec4 posMassOther = texelFetch(sTD2DInputs[0], ivec2(i,j), 0);
					vec4 posMassOther = posMassShared[k][l];
					vec2 posOther = posMassOther.xyz;
					float massOther = posMassOther.w;
					
					vec2 dir = pos - posOther;
					float dist = length(dir);
					
					if(dist>0)

					////
				
					if(dot(normx(dir), mydirection) > angle) // sensing angle
					{
						vec4 velEspeceOther = velEspeceShared[k][l];
						vec2 velOther = posMassOther.xyz;
						float especeOther = posMassOther.w;
						
						if(velOther.r != 0 && velOther.g != 0 && velOther.b != 0 && especeOther == 1)
						{
							avg += velOther.rgb;
							total ++;
						}

					}

		    	}
		    }
		    barrier();
    	}
    	barrier();
    }
    	
	if (total > 0)
	{
		avg /= total; // number of neighbors verified
		avg = normx(avg);
		avg *= uSteerConfigs.y;
		steer = avg - b.speed;
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

	vec3 steer = vec3(0);

	vec3 mydirection = normx(b.speed);

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
					

					if(dot(normx(direction), mydirection) > angle) // sensing angle
					{
						vec3 coord = vec3(b.position + direction);
						vec3 neighborVelocity = texelFetch(sTD2DInputs[INOUT_DRAW_SPEED_HEALTH_BUFFER], ivec2(coord), 0).rgb;
						float neighborEspece = texelFetch(sTD2DInputs[INOUT_COLOR_AGENTS_BUFFER], ivec2(coord), 0).r;

						
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
		steer *= uSteerConfigs.y;
		steer -= b.speed;
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

	vec3 steer = vec3(0);

	vec3 mydirection = normx(b.speed);

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
					

					if(dot(normx(direction), mydirection) > angle) // sensing angle
					{
						vec3 coord = vec3(b.position + direction);
						vec3 neighborVelocity = texelFetch(sTD2DInputs[INOUT_DRAW_SPEED_HEALTH_BUFFER], ivec2(coord), 0).rgb;
						float neighborEspece = texelFetch(sTD2DInputs[INOUT_COLOR_AGENTS_BUFFER], ivec2(coord), 0).r;
						
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
		steer *= uSteerConfigs.y;
		steer -= b.speed;
	}

	return steer;
}

// UPDATE AGENTS

void UpdateAgent(inout Agent agent) 
{
	if(AliveAgent(agent))
	{
		// Update speed and direction
		agent.speed += agent.acceleration;

		// Add some random noise to agent's speed
		vec3 noise = vec3(random2(uTime * 0.1975 * Index(agent.myCoord)), random2(uTime * 0.75 * Index(agent.myCoord)), random2(Index(agent.myCoord) * uTime * 0.15)) * 2.0 - 1.0; // Assume random2() returns a random number between -1 and 1.
		agent.speed += noise * 0.0; // Tune the 0.1 factor to increase or decrease the noise intensity

		float speed = length(agent.speed);
		vec3 direction = normalize(agent.speed);

		speed = clamp(speed, agent.minSpeed, agent.maxSpeed);

		agent.speed = direction * speed;
		agent.direction = direction;
		agent.position += agent.speed;
		

		agent.health -= random2(agent.size + agent.position.x * agent.position.y * uTime * 0.191) * uDecreaseIncreaseHealthRate.x;
			
		float size = parabola(1.0 - agent.health, 1.0);
		agent.size =  Map(size, 0, 1, 0.3, 3);

		// Boundary wrapping
		agent.position.x = mod(agent.position.x + uRes.x, uRes.x);
		agent.position.y = mod(agent.position.y + uRes.x, uRes.x);
		agent.position.z = mod(agent.position.z + uRes.y, uRes.y);

		// Add a damping factor to the acceleration instead of setting it to zero.
		agent.acceleration *= 0.9; // You may adjust this damping factor as needed.
	}
	else
	{
		agent.position = vec3(-999999);
		agent.speed = vec3(0.0);
		agent.acceleration = vec3(0.0);
		agent.size = 0.0;
		agent.mass = 0.0;
		agent.maxSpeed = 0.0;
		agent.minSpeed = 0.0;
		agent.direction = vec3(0,0,0);
		agent.health = 0.0;
		agent.DNA = vec4(10.0);
		agent.DNAScaled = vec4(3.0);
		agent.color = vec4(0.0);
		agent.espece = 0;
		agent.myCoord = ivec2(-9999);
	}
	
}

// WRITE DATA INTO TEXTURES

void Write (Agent a, Food foods[7], ivec2 coord, int closest)
{
	vec4 outDrawMassCoord = vec4(0);
	vec4 outDrawSpeedHealth = vec4(0);

    // Check if agent's position is within the boundaries
    bool withinBoundaries = a.position.x >= 0 && a.position.x <= uRes.x && a.position.y >= 0 && a.position.y <= uRes.x;

    if (AliveAgent(a)) 
    {
		imageStore(mTDComputeOutputs[INOUT_POSITION_AND_MASS_BUFFER], coord, TDOutputSwizzle(vec4(a.position, a.mass)));
		imageStore(mTDComputeOutputs[INOUT_VELOCITY_AND_SIZE_BUFFER], coord, TDOutputSwizzle(vec4(a.speed, a.size)));
		imageStore(mTDComputeOutputs[INOUT_DIRECTION_AND_HEALTH_BUFFER], coord, TDOutputSwizzle(vec4(a.direction, a.health)));
		imageStore(mTDComputeOutputs[INOUT_DNA_BUFFER], coord, TDOutputSwizzle(a.DNA));
		imageStore(mTDComputeOutputs[INOUT_COLOR_AGENTS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(a.color)));
		outDrawMassCoord = vec4(a.mass, a.myCoord, a.color.r);
		outDrawSpeedHealth = vec4(a.speed, a.health);
		imageStore(mTDComputeOutputs[INOUT_DRAW_MASS_COORD_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(outDrawMassCoord));
		imageStore(mTDComputeOutputs[INOUT_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(outDrawSpeedHealth));
		imageStore(mTDComputeOutputs[INOUT_MEMORYFOOD_COLOR_B_BUFFER], coord, TDOutputSwizzle(vec4(a.foodMemory, a.color.b)));
		imageStore(mTDComputeOutputs[OUT_ORIENTATIONS_BUFFER], coord, TDOutputSwizzle(vec4(a.color.rgb, a.orientation)));
    }

	else
	{
		imageStore(mTDComputeOutputs[INOUT_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[INOUT_DRAW_MASS_COORD_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[INOUT_COLOR_AGENTS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[INOUT_DIRECTION_AND_HEALTH_BUFFER], coord, TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[INOUT_DNA_BUFFER], coord, TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[INOUT_VELOCITY_AND_SIZE_BUFFER], coord, TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[INOUT_MEMORYFOOD_COLOR_B_BUFFER], coord, TDOutputSwizzle(vec4(uRes.x * 0.5, uRes.x * 0.5, uRes.y * 0.5, 9999)));
		imageStore(mTDComputeOutputs[INOUT_POSITION_AND_MASS_BUFFER], coord, TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[OUT_TRAILS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
		imageStore(mTDComputeOutputs[OUT_ORIENTATIONS_BUFFER], coord, TDOutputSwizzle(vec4(0.0, 0.0, 0.0, 0.0)));

	}
	
	// Check if food's position is within the boundaries
    bool foodWithinBoundaries = coord.x < uCounts.y && coord.y == 0;

    if (foodWithinBoundaries) 
    {
        if(coord.x < uCounts.y && coord.y == 0)
        {
            vec4 foodColor = vec4(foods[coord.x].foodColor.rgb, 1.0);
            imageStore(mTDComputeOutputs[INOUT_DRAW_FOOD_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
            imageStore(mTDComputeOutputs[INOUT_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
            imageStore(mTDComputeOutputs[INOUT_DRAW_MASS_COORD_BUFFER], ivec2(round(foods[coord.x].position.xy)), TDOutputSwizzle(foodColor));
        }
    }

}

void WriteDiffuseTexture(Agent a, ivec2 posOnBuffer)
{
    float uTrailReinforcementFactor = 0.01;

    // color of present pixel
    vec4 oc = texelFetch(sTD2DInputs[INOUT_COLOR_AGENTS_BUFFER], posOnBuffer, 0);
    vec4 colorPresentPixel = texelFetch(sTD2DInputs[IN_TRAILS_BUFFER], posOnBuffer, 0);

    // Check if the color is close to the color of species 2 (r = 0.7, g = 0.0, b = 1.0)
    bool isSpecies2 = abs(colorPresentPixel.r - 0.7) < 0.05 && abs(colorPresentPixel.g - 0.0) < 0.05 && abs(colorPresentPixel.b - 1.0) < 0.05;

    // Decay the color by a certain factor if it's not species 2
    if (!isSpecies2 && colorPresentPixel.a > 0.01) 
    {
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
            float especeNeighbor = texelFetch(sTD2DInputs[INOUT_COLOR_AGENTS_BUFFER], coord, 0).r;

            // average it
            if (especeNeighbor == 0.7)
            {
                avg += texelFetch(sTD2DInputs[IN_TRAILS_BUFFER], coord, 0).r;
            }
        }
    }

    // multiply for trail decay factor
    avg /= 9;

    // Reinforce existing trails only for species 2
    if(isSpecies2)
    {
        avg += uTrailReinforcementFactor; // Increase the trail level by a small amount at each time step
    }

    oc += vec4(avg * uDecayFactorAnts);
    oc = clamp(oc, 0, 1);

    // update
    imageStore(mTDComputeOutputs[OUT_TRAILS_BUFFER], posOnBuffer, TDOutputSwizzle(colorPresentPixel + oc));
}

// MAIN FUNCTION

void main()
{
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

	// AGENT DATA
	Agent agent = ReadAgent(coord); // read present coord agent data

	if (ActiveAgent(coord)) // if agent exists
	{
		if(uResetTextureAndAgents == 1)
		{
			agent = ResetAgents(coord);
		}

		if (AliveAgent(agent))
		{
			if(uResetTextureAndAgents == 1)
			{
				agent = ResetAgents(coord);
			}

			if(agent.espece == 1 ) // BOIDS
			{
	
				vec3 alignForce = Align(agent);
				vec3 cohesionForce = Cohesion(agent) ;
				vec3 separationForce = Separation(agent) ;

				ApplyForce(agent, alignForce + cohesionForce + separationForce);
				
			}

			else if(agent.espece == 2) // PHYSARUM
			{
				vec3 antsTurns = AntPhysarumBehavior(agent, DNA1scaled, DNA0scaled);

				ApplyForce(agent, antsTurns);				
			}

			else if (agent.espece == 3) // PP
			{
				ApplyOrientationForcePPSAgent(agent, 1.0);
			}

			UpdateAgent(agent);
		}
			
		else // if agent is dead
		{
			
			agent = GenerateEmptyAgent();
			
		}

	}

	else // if agent doesn't exist, out the max agents list
	{
		agent = GenerateEmptyAgent();
	}

	WriteDiffuseTexture(agent, coord);
	Write(agent, foods, coord, closest);
	
}
			
