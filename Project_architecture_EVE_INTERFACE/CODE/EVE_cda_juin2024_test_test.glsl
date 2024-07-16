uniform float uTime;
uniform vec3 uRes;                      // its a square with x and y bounds of 512px (uRes.x) and the z bound is two (uRes.y)
uniform vec2 uCounts;                   // uAgentCount, uFoodCount
uniform int uResetTextureAndAgents;
uniform vec3 uSteerConfigs;             // uVelDamping (NOT USING), uMaxSpeed, uAbsMaxSteerForce
uniform int uAttractionConfigs;         // uAttractionForce
uniform ivec4 uRangeNeighbors;          // uRangePPS, uRangeFlock, uRangeAnts, uRangeReproduction
uniform float uG;                       // (NOT USING)
uniform vec2 uConfigsAnts;              // uSensAngleAnts, uMaxTrailLimits
uniform vec3 uFlockConfigs;             // uAlign, uCohesion, uSeparation
uniform vec3 uSeekSettings;             // uRad to seek attractors, uAngleFlock, seekFoodForce
uniform float uDecayFactorAnts;         // 
uniform vec4 uNoiseFactor;              // 
uniform float uDelta;                   // 
uniform float uMutationRate;
uniform float uReproductionChance;
uniform int uNumAttractors;
uniform int uDNA;
uniform float uAvoidDistance;           // (NOT USING)
uniform float uAvoidBoidsFactor;        // (NOT USING)
uniform float uBalanceFactor;           // (NOT USING function)
uniform float uAlphaPPS;                // 
uniform float uBetaPPS;                 // 
uniform int uRadiusPPS;                 // (NOT USING)
uniform float uAntRadians2;             // rotation angle that we rotate by for turns
uniform int uNoiseOn;                   // Add noise to speed
uniform vec2 uOrigin;
uniform vec2 uSizeObstacle;             // size of obstacle

uniform vec2 uDecreaseIncreaseHealthRate;

uniform samplerBuffer sFoodPositions;

#define PI 3.14159265358979323846
#define HALFPI 1.57079632679489661923
#define TWOPI 6.283185307179586
#define deg2rad 0.01745329251

#define IN_INIT_POS 0
#define OUT_TRAILS_BUFFER 0
#define INOUT_POSITION_AND_MASS_BUFFER 1
#define INOUT_VELOCITY_AND_SIZE_BUFFER 2
#define INOUT_DIRECTION_AND_HEALTH_BUFFER 3
#define INOUT_DNA_BUFFER 4
#define INOUT_COLOR_AGENTS_BUFFER 5

#define INOUT_MEMORYFOOD_COLOR_B_BUFFER 6

#define INOUT_DRAW_FOOD_BUFFER 7
#define INOUT_DRAW_MASS_COORD_BUFFER 8
#define INOUT_DRAW_SPEED_HEALTH_BUFFER 9
#define IN_TRAILS_BUFFER 10

#define IN_ATTRACTORS_BUFFER 11

#define OUT_ORIENTATIONS_BUFFER 11
#define IN_ORIENTATIONS_BUFFER 12
#define OUT_POS_SPECIES_BUFFER 12

layout (local_size_x = 8, local_size_y = 8) in;

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
    float maxSpeed;
    float minSpeed;
    vec3 acceleration;
    //vec3 rememberedFoodLocation;
};

struct Food
{
    vec3 position;
    vec4 foodColor;
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

vec3 map_vector(vec3 vector, float min_val, float max_val) 
{
    float vector_range = max(vector.r, max(vector.g, vector.b)) - min(vector.r, min(vector.g, vector.b));
    vec3 mapped_vector = (vector - vec3(min(vector.r, min(vector.g, vector.b)))) / vector_range * (max_val - min_val) + vec3(min_val);
    return mapped_vector;
}

vec3 hsv2rgb(float h, float s, float v) 
{
    vec3 c = vec3(h, s, v);
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec2 hsv2gb(float h, float s, float v) 
{
    vec3 rgb = hsv2rgb(h, s, v);
    return vec2(rgb.g, rgb.b);
}

// AGENT UPDATE AND CHECKING

int Index(ivec2 coord) 
{
    ivec2 clampedCoord = clamp(coord, ivec2(0), ivec2(uRes.x - 1, uRes.y - 1));
    int index = clampedCoord.x + clampedCoord.y * int(uRes.x);
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

    a.acceleration = vec3(0.0);
    a.position = posMass.rgb;
    a.mass = posMass.a;
    if(a.mass <= 0.1){a.mass = 0.2;}
    a.speed = velSize.rgb;
    a.size = velSize.a;
    a.direction = dirHealth.rgb;
    a.DNA = dna;
    a.myCoord = agentCoord;  

    a.maxSpeed = Map(a.DNA.b, 0, 1, 0.3 + uSteerConfigs.y, 2 + uSteerConfigs.y); // Adding 0.5 to make sure maxSpeed is always greater than 0.5
    a.minSpeed = a.DNA.b; // Reducing the multiplication factor to make sure minSpeed is always less than maxSpeed

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
        a.color.a = a.health;
        vec4 orientationColorIN = texelFetch(sTD2DInputs[IN_ORIENTATIONS_BUFFER], coord, 0);
        a.orientation = orientationColorIN.a;
        vec2 gbColor = vec2(orientationColorIN.g, orientationColorIN.b);
        a.color.gb = gbColor; // using green and blue channels for color
    }

    return a;
}


void ReadFoods(inout Food foods[13])
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

    return agent;
}

// RESET AGENTS

Agent ResetAgents(ivec2 posOnTexture)
{
    float seed = random2(float(posOnTexture.x * posOnTexture.y) * fract(uTime) * 0.05);
    vec3 center = vec3(uOrigin, uRes.z*0.5);
    float angle = PI * 2 * random2(seed * uTime);

    vec3 posRandom = vec3(
        max(1.0, random2(float(posOnTexture.x * posOnTexture.y) * fract(uTime) * 5) * uRes.x),
        max(1.0, random2(float(posOnTexture.x + posOnTexture.y) * fract(uTime) * 9) * uRes.y),
        max(1.0, random2(float(posOnTexture.y) * fract(uTime) * 15) * uRes.z)
    );

    vec3 velRandom = vec3(
        random2(float(posOnTexture.x + 10) + fract(uTime)), 
        random2(float(posOnTexture.y + 0.5) + fract(uTime) * 0.715), 
        random2(float(posOnTexture.x) * fract(uTime) * 0.915)
    ) * 2.0 - 1.0;

    vec4 dna = vec4(
        random2(float(posOnTexture.x) * fract(uTime) * 20.905), 
        random2(float(posOnTexture.y) * fract(uTime) * 9.001), 
        random2(float(posOnTexture.x) * fract(uTime) * 0.715), 
        random2(float(posOnTexture.x) * fract(uTime) * 9.905)
    );

    Agent a;
    a.position = posRandom;
    a.speed = velRandom;

    vec3 random_direction = vec3(
        random2(float(posOnTexture.x) + uTime * 0.4), 
        random2(float(posOnTexture.y + 0.7) + uTime * 0.915), 
        random2(float(posOnTexture.x) + uTime * 0.315)
    );

    a.direction = normalize(random_direction) * 2.0 - 1.0;
    a.acceleration = vec3(0);
    a.DNA = dna;
    a.health = 2 * Map(random2(a.size + a.position.x * a.position.y * uTime * 0.191), 0.0, 1.0, 1.0, 3.0);
    a.maxSpeed = Map(a.DNA.b, 0, 1, 0.3 + uSteerConfigs.y, 2 + uSteerConfigs.y);
    a.minSpeed = a.DNA.b;
    a.mass = a.DNA.g * 3;
    a.size = a.DNA.b * 3;
    a.myCoord = posOnTexture;
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
        a.color = vec4(0.4, 1.0, 0.0, a.health);
        a.orientation = random2(float(posOnTexture.x + posOnTexture.y) * fract(uTime) * 0.005) * TWOPI;
    }

    return a;
}


// REPRODUCTION AGENTS

int getRandomInt(vec2 seed) 
{
    // Create a pseudo-random number from the sine of the dot product
    // The randomness of the number is based on the seed value
    float rand = fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);
    
    // Scale the random number to be between 0 and 6
    int result = int(rand * 7.0);
    
    return result;
}

Agent crossoverAndMutate(ivec2 posOnTexture, Agent Myself, ivec2 mateCoord) 
{
    Agent child;
    child = ResetAgents(posOnTexture);
    Agent mate = ReadAgent(mateCoord);

    int midpoint = getRandomInt(vec2(0, 4)); // Now the midpoint is an integer between 0 and 3.

    //Crossover
    child.DNA.r = midpoint > 0 ? mate.DNA.r : Myself.DNA.r;
    child.DNA.g = midpoint > 1 ? mate.DNA.g : Myself.DNA.g;
    child.DNA.b = midpoint > 2 ? mate.DNA.b : Myself.DNA.b;
    child.DNA.a = midpoint > 3 ? mate.DNA.b : Myself.DNA.b;

    //Mutate
    float mutationRate = uMutationRate;

    for(int i = 0; i < 4; i++) 
    {
        float mutationChance = random2(Index(posOnTexture) * uTime * 0.921);

        if(mutationChance < mutationRate) 
        {           
            if(i == 0) 
            {
                child.DNA.r = clamp(random2(posOnTexture.x * uTime + i * 0.01), 0.0, 1.0);
            }
            else if(i == 1) 
            {
                child.DNA.g = clamp(random2(posOnTexture.y * uTime + i * 0.01), 0.0, 1.0);
            }
            else if(i == 2) 
            {
                child.DNA.b = clamp(random2(posOnTexture.x * uTime + i * 0.01), 0.0, 1.0);
            }
            else if(i == 3) 
            {
                child.DNA.a = clamp(random2(posOnTexture.y * uTime + i * 0.01), 0.0, 1.0);
            }
        }
    }
    // Keeping the species mutation outside the loop
    //child.DNA.a = 0.7;

    // Assigning these properties outside the loop to avoid constant overwriting
    if(midpoint > 2) // Assuming midpoint is the determining factor for inheriting properties
    {
        child.position = Myself.position + 2.0;
        //child.color = a1.color;
        child.maxSpeed = Myself.maxSpeed;
    }
    else
    {
        child.position = mate.position - 2.0;
        //child.color = a2.color;
        child.maxSpeed = mate.maxSpeed;
    }

    if(child.DNA.a <= 0.3)
    {
        child.espece = 1;
        child.color = vec4(1.0, 0, 0, child.health);
    }
    else if (child.DNA.a <= 0.7)
    {
        child.espece = 2;
        child.color = vec4(0.7, 0, 1.0, child.health);
    }
    else if (child.DNA.a <= 1)
    {
        child.espece = 3;
        child.color = vec4(0.4, mate.color.g, Myself.color.b, child.health); 
    }

    child.health = 6;

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
    float newMaxSpeed = a.maxSpeed + Map(a.DNA.b, 0, 1, 0.5 * uSteerConfigs.y, 2 * uSteerConfigs.y);

    if(dist < rad/2) // very close to the target
    {
        desired *= Map(dist, 0, rad/2, 0, newMaxSpeed);
        vec3 steer = desired - a.speed;
        steer = clamp(steer, vec3(-newMaxSpeed), vec3(newMaxSpeed));
        ApplyForce(a, steer * mag);
    }
    else if(dist < rad) // close to the target
    {
        desired *= Map(dist, 0, rad, 0, newMaxSpeed);
        vec3 steer = desired - a.speed;
        steer = clamp(steer, vec3(-newMaxSpeed), vec3(newMaxSpeed));
        ApplyForce(a, steer * mag);
    }
    else if(dist < rad * 2 && dist >= rad) // within double the target radius
    {
        desired *= newMaxSpeed;
        vec3 steer = desired - a.speed;
        steer = clamp(steer, vec3(-newMaxSpeed), vec3(newMaxSpeed));
        ApplyForce(a, steer * mag);
    }
    else
    {
        return;
    }
}

float pointSquareDistance(vec2 P, vec2 origin, float s) 
{
    vec2 minCorner = origin - vec2(s, s);
    vec2 maxCorner = origin + vec2(s, s);

    // Clamp point within the square's extents to get the closest point on or inside the square
    vec2 closestPoint = clamp(P, minCorner, maxCorner);
    
    return length(P - closestPoint);
}

// Simple hashing function to produce noise
float hash(vec2 p) 
{
    vec3 p3  = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float pointRectDistanceWithNoise(vec2 P, vec2 origin, float halfWidth, float halfHeight) 
{
    vec2 minCorner = origin - vec2(halfWidth, halfHeight);
    vec2 maxCorner = origin + vec2(halfWidth, halfHeight);

    // Clamp point within the rectangle's extents to get the closest point on or inside the rectangle
    vec2 closestPoint = clamp(P, minCorner, maxCorner);
    
    float distance = length(P - closestPoint);

    // Add noise to the distance
    float noiseAmount = 0.05; // Adjust as necessary
    distance += noiseAmount * (hash(P * 10.0) - 0.5); // Multiplied P by 10.0 to increase noise frequency

    return distance;
}

void Attractors(inout Agent a)
{
    Agent b = a;

    float distParticipant = Map(b.DNA.b + b.DNA.g, 0, 2, 1, 20);

    for(int i = 0; i <= uNumAttractors; i ++)
    {
        vec4 attractor = texelFetch(sTD2DInputs[IN_ATTRACTORS_BUFFER], ivec2(i, 0), 0);
        vec3 attractorPos = attractor.rgb;
        
        vec2 origin = attractorPos.xy;
        float halfWidth = uSizeObstacle.x;
        float halfHeight = uSizeObstacle.y;

        vec2 P = b.position.xy; 
        float dist = pointRectDistanceWithNoise(P, origin, halfWidth, halfHeight);

        if (dist < distParticipant)
        {
            a.health = 0.0;
        }
        else
        {
            // Use species-dependent behavior
            switch(a.espece) 
            {
                case 1: // species 1
                    SeekTarget(a, attractor.xyz, uAttractionConfigs * 10.0, distParticipant);
                    break;
                case 2: // species 2
                    SeekTarget(a, attractor.xyz, uAttractionConfigs * 5.0, distParticipant);
                    break;
                case 3: // species 3
                    SeekTarget(a, attractor.xyz, uAttractionConfigs * 8.0, distParticipant);
                    break;
            }
        }
    }
}

vec3 Eat(Agent a, Food foods[13])
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
    a.foodMemory = foodPos;    
}

// PPS DNA CODE
bool isOnRightSide(vec2 p, vec2 v, vec2 op) // position, speed and other position
{
    vec2 b = p + v;
    return ((b.x - p.x) * (op.y - p.y) - (b.y - p.y) * (op.x - p.x)) > 0.0;
}

vec2 updateLandR(Agent b)
{
    float left = 0.0;
    float right = 0.0;

    // int rangeNeighbors = int(Map(b.DNA.r, 0, 1, uRangeNeighbors.x-2, uRangeNeighbors.x+2));

    int rangeNeighbors = int(uRangeNeighbors.x);

    for (int x = -rangeNeighbors; x <= rangeNeighbors; x++)
    {
        for(int y = -rangeNeighbors; y <= rangeNeighbors; y++)
        {
            if (!(x == 0 && y == 0)) // NOT CURRENT CELL
            {
                vec2 targetPos = vec2(b.position.x + x, b.position.y + y);
                vec3 targetSpeciesAndColor = texelFetch(sTD2DInputs[INOUT_COLOR_AGENTS_BUFFER], ivec2(targetPos), 0).rba;
                float targetHealth = targetSpeciesAndColor.z;
                float targetSpecies = targetSpeciesAndColor.x;

                if(targetHealth > 0 && targetSpecies == 0.4)
                {
                    if (distance(b.position.xy, targetPos) < rangeNeighbors) 
                    {
                        if (isOnRightSide(b.position.xy, b.speed.xy, targetPos)) 
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
    }

    return vec2(right, left);
}

void ApplyOrientationForcePPSAgent(inout Agent agent, float alpha, float beta)
{
    vec2 rl = updateLandR(agent);
    float r = rl.x;
    float l = rl.y;
    float n = r + l;

    float delta_phi = alpha + beta * n * sign(r - l);

    agent.orientation += delta_phi;

    vec3 direction = vec3(cos(agent.orientation), sin(agent.orientation), 0.0);

    agent.acceleration = direction * agent.maxSpeed;

    // Use n to influence both the green and blue channels
    float h = 0.5; // Keep hue constant
    float s = 1.0; // Full saturation
    float v = clamp(n / 20.0, 0.0, 1.0); // Use n to vary the brightness

    vec2 gbColor = hsv2gb(h, s, v);
    agent.color.g = gbColor.x;
    agent.color.b = gbColor.y;
}


// UPDATE AGENTS

void UpdateAgent(inout Agent agent)
{
    if(AliveAgent(agent))
    {
        if(agent.espece == 3)
        {
            // Use angle-based velocity update
            vec3 direction = vec3(cos(agent.orientation), sin(agent.orientation), 0.0);
            agent.speed = direction * agent.maxSpeed;

            agent.position += agent.speed;

            agent.health -= random2(agent.size * uTime * 0.191) * uDecreaseIncreaseHealthRate.x;

            float size = parabola(1.0 - agent.health, 2.0);
            agent.size = Map(size, 0, 1, 0.3, 3);

            agent.position.x = mod(agent.position.x + uRes.x, uRes.x);
            agent.position.y = mod(agent.position.y + uRes.y, uRes.y);
            agent.position.z = mod(agent.position.z + uRes.z, uRes.z);

            agent.acceleration *= 0.9;

            if (agent.position.x == 0 && agent.position.y == 0)
            {
                agent.health = 0;
            }
        }
        else
        {
            // Update speed and direction
            agent.speed += agent.acceleration;
            float speed = length(agent.speed);
            vec3 direction = normalize(agent.speed);
            speed = clamp(speed, agent.minSpeed + uSteerConfigs.y*0.3, agent.maxSpeed + uSteerConfigs.y*2);
            agent.speed = direction * speed;
            agent.direction = direction;
            agent.position += agent.speed;
        }

        agent.health -= random2(agent.size * uTime * 0.191) * uDecreaseIncreaseHealthRate.x;

        float size = parabola(1.0 - agent.health, 2.0);
        agent.size = Map(size, 0, 1, 0.3, 3);

        // Boundary wrapping
        agent.position.x = mod(agent.position.x + uRes.x, uRes.x);
        agent.position.y = mod(agent.position.y + uRes.y, uRes.y);
        agent.position.z = mod(agent.position.z + uRes.z, uRes.z);

        // Add a damping factor to the acceleration instead of setting it to zero.
        agent.acceleration *= 0.9; // You may adjust this damping factor as needed.

        if(agent.position.x == 0 && agent.position.y == 0)
        {
            agent.health = 0;
        }
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
        agent.color = vec4(0.0);
        agent.espece = 0;
        agent.myCoord = ivec2(-9999);
    }
}


// WRITE DATA INTO TEXTURES

void Write (Agent a, Food foods[13], ivec2 coord, int closest)
{
    vec4 outDrawMassCoord = vec4(0);
    vec4 outDrawSpeedHealth = vec4(0);

    // Check if agent's position is within the boundaries
    bool withinBoundaries = a.position.x >= 0 && a.position.x <= uRes.x && a.position.y >= 0 && a.position.y <= uRes.y;

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

        if(a.espece == 3)
        {
            // Calculate n value
            vec2 rl = updateLandR(a);
            float r = rl.x;
            float l = rl.y;
            float n = r + l;

            // Normalize n to [0, 1] range for coloring
            float normalizedN = clamp(n / 20.0, 0.0, 1.0);

            // Store in OUT_ORIENTATIONS_BUFFER with n in the red, green, and blue channels
            imageStore(mTDComputeOutputs[OUT_ORIENTATIONS_BUFFER], coord, TDOutputSwizzle(vec4(normalizedN, normalizedN, normalizedN, a.orientation)));
        }
        else
        {
            imageStore(mTDComputeOutputs[OUT_ORIENTATIONS_BUFFER], coord, TDOutputSwizzle(vec4(a.color.rgb, a.orientation)));
        }

        imageStore(mTDComputeOutputs[OUT_POS_SPECIES_BUFFER], coord, TDOutputSwizzle(vec4(a.position, a.espece)));
    }
    else
    {
        imageStore(mTDComputeOutputs[INOUT_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
        imageStore(mTDComputeOutputs[INOUT_DRAW_MASS_COORD_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
        imageStore(mTDComputeOutputs[INOUT_COLOR_AGENTS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
        imageStore(mTDComputeOutputs[INOUT_DIRECTION_AND_HEALTH_BUFFER], coord, TDOutputSwizzle(vec4(0)));
        imageStore(mTDComputeOutputs[INOUT_DNA_BUFFER], coord, TDOutputSwizzle(vec4(0)));
        imageStore(mTDComputeOutputs[INOUT_VELOCITY_AND_SIZE_BUFFER], coord, TDOutputSwizzle(vec4(0)));
        imageStore(mTDComputeOutputs[INOUT_MEMORYFOOD_COLOR_B_BUFFER], coord, TDOutputSwizzle(vec4(uRes.x * 0.5, uRes.y * 0.5, uRes.z * 0.5, 9999)));
        imageStore(mTDComputeOutputs[INOUT_POSITION_AND_MASS_BUFFER], coord, TDOutputSwizzle(vec4(0)));
        imageStore(mTDComputeOutputs[OUT_TRAILS_BUFFER], ivec2(round(a.position.xy)), TDOutputSwizzle(vec4(0)));
        imageStore(mTDComputeOutputs[OUT_ORIENTATIONS_BUFFER], coord, TDOutputSwizzle(vec4(0.0, 0.0, 0.0, 0.0)));
        imageStore(mTDComputeOutputs[OUT_POS_SPECIES_BUFFER], coord, TDOutputSwizzle(vec4(0.0)));
    }

    // Check if food's position is within the boundaries
    
    for(int i = 0; i < uCounts.y; i++)
    {
        vec4 foodInfo = texelFetch(sFoodPositions, i);
        foods[i].position.xyz =  foodInfo.rgb;
        foods[i].foodColor =  foodInfo;
        foods[i].index = i;
        imageStore(mTDComputeOutputs[INOUT_DRAW_FOOD_BUFFER], ivec2(round(foods[i].position.xy)), TDOutputSwizzle(foods[i].foodColor));
        imageStore(mTDComputeOutputs[INOUT_DRAW_SPEED_HEALTH_BUFFER], ivec2(round(foods[i].position.xy)), TDOutputSwizzle(foods[i].foodColor));
        imageStore(mTDComputeOutputs[INOUT_DRAW_MASS_COORD_BUFFER], ivec2(round(foods[i].position.xy)), TDOutputSwizzle(vec4(1,0,0,1)));
    }
}


void WriteTrailsTexture(Agent a, ivec2 posOnBuffer)
{
    float uTrailReinforcementFactor = uConfigsAnts.y;

    // color of present pixel in the trail buffer
    vec4 colorPresentPixel = texelFetch(sTD2DInputs[IN_TRAILS_BUFFER], posOnBuffer, 0);

    // get the color of the agent in the agents buffer
    vec4 colorAgent = texelFetch(sTD2DInputs[INOUT_COLOR_AGENTS_BUFFER], posOnBuffer, 0);

    // check if the agent is of species 2 based on the red channel of the agent's color
    bool isSpecies2 = abs(colorAgent.r - 0.7) < 0.05;

    // Decay the color by a certain factor for every pixel
    colorPresentPixel *= uDecayFactorAnts;
    colorPresentPixel = clamp(colorPresentPixel, 0, 1);

    if(isSpecies2)
    {
        float avg = 0;

        // look at surrounding 9 squares
        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                // surrounding square coordinate
                float coordX = mod(posOnBuffer.x + x + uRes.x, uRes.x);
                float coordY = mod(posOnBuffer.y + y + uRes.y, uRes.y);

                vec2 coord = vec2(coordX, coordY);

                // trail level of the neighbor
                float trailNeighbor = texelFetch(sTD2DInputs[IN_TRAILS_BUFFER], ivec2(coord), 0).b;

                // add the neighbor's trail level to the average
                avg += trailNeighbor;
            }
        }

        // calculate the average trail level
        avg = 0.9 * (avg/9);

        // reinforce the trail level
        avg += uTrailReinforcementFactor;

        // Add the average trail level to the current trail level, creating a blur effect
        colorPresentPixel.b = max(avg, colorPresentPixel.b); // use max instead of simple addition
    }
    else
    {
        // for other species, add some fraction of the average of the surrounding squares, to create a blur effect
        float avg = 0;

        // look at surrounding 9 squares
        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                // surrounding square coordinate
                float coordX = mod(posOnBuffer.x + x + uRes.x, uRes.x);
                float coordY = mod(posOnBuffer.y + y + uRes.y, uRes.y);

                vec2 coord = vec2(coordX, coordY);

                // trail level of the neighbor
                float trailNeighbor = texelFetch(sTD2DInputs[IN_TRAILS_BUFFER], ivec2(coord), 0).b;

                // add the neighbor's trail level to the average
                avg += trailNeighbor;
            }
        }

        // calculate the average trail level
        avg /= 9;

        // add the average to the current trail level
        colorPresentPixel.b += avg * 0.01; // you can adjust the 0.1 factor to control the amount of blurring
    }

    // clamp the color value between 0 and 1
    colorPresentPixel = clamp(colorPresentPixel, 0, 1);

    // write the updated trail level to the output
    imageStore(mTDComputeOutputs[OUT_TRAILS_BUFFER], posOnBuffer, TDOutputSwizzle(colorPresentPixel));
}


ivec2 findEmptyCell(ivec2 coord) 
{
    int maxAttempts = 10;  // define a limit to avoid infinite loops
    int attempt = 0;
    ivec2 potentialCell;

    while(attempt < maxAttempts)
    {
        int randomX = int(randomVec4Range(coord, uTime * 0.58, 0.0, float(uRes.x)).x);
        int randomY = int(randomVec4Range(coord, uTime * 25.21, 0.0, float(uRes.y)).y);
        
        potentialCell = ivec2(randomX, randomY);

        if (ActiveAgent(potentialCell))
        {
            Agent potentialAgent = ReadAgent(potentialCell);

            if(potentialAgent.health <= 0.0)
            {
                return potentialCell;
            }
        } 
        attempt++;
    }

    return ivec2(-99);  // return a non-valid coordinate if no empty cell found after maxAttempts
}

// MAIN FUNCTION
void main()
{
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    Agent agent = ReadAgent(coord); 
    Agent mate = GenerateEmptyAgent();
    ivec2 mateCoord = ivec2(-99); 

    bool validMateFound = false;

    Food foods[13];

    ReadFoods(foods); 
    int closest = -1;

    if (ActiveAgent(coord))
    {
        if (uResetTextureAndAgents == 1)
        {
            agent = ResetAgents(coord);
        }

        if (AliveAgent(agent))
        {
            if (uResetTextureAndAgents == 1)
            {
                agent = ResetAgents(coord);
            }

            vec3 foodTarget = vec3(0.0);
            float useRememberedProb = agent.DNA.b;

            vec3 closestAndRecord = Eat(agent, foods);
            int closestFoodIndex = int(round(closestAndRecord.x));

            if (closestFoodIndex != -1) 
            {
                foodTarget = foods[closestFoodIndex].position;
                RememberFood(agent, foodTarget);
            }
            
            if (agent.health < 5)
            {
                if(length(foodTarget) > 0)
                {
                    float maxSteerForce = uSteerConfigs.z;
                    float seekfood = uSeekSettings.z;

                    SeekTarget(agent, foodTarget, maxSteerForce * 10, seekfood);

                    if(distance(agent.position, foodTarget) < 1)
                    {
                        agent.health += uDecreaseIncreaseHealthRate.y;
                    }
                }
            }

            else if (agent.health < 10)
            {
                if (randomVec4Range(agent.myCoord, uTime, 0.0, 1.0).x < useRememberedProb && length(agent.foodMemory) > 0) 
                {
                    foodTarget = agent.foodMemory;
                }

                if(length(foodTarget) > 0)
                {
                    float maxSteerForce = uSteerConfigs.z;
                    float seekfood = uSeekSettings.z;

                    SeekTarget(agent, foodTarget, maxSteerForce * 3, seekfood);

                    if(distance(agent.position, foodTarget) < 1)
                    {
                        agent.health += uDecreaseIncreaseHealthRate.y;
                    }
                }
            }

            if(agent.espece == 3) // PPS-specific logic
            {
                ApplyOrientationForcePPSAgent(agent, radians(uAlphaPPS), radians(uBetaPPS));
            }

            if(uNumAttractors >= 1)
            {
                Attractors(agent);
            }
            
            if(agent.health >= 4)
            {
                float healthRecord = 0;

                int rangeNeighbors = int(uRangeNeighbors.a);

                for (int x = -rangeNeighbors; x <= rangeNeighbors; x++)
                {
                    for(int y = -rangeNeighbors; y <= rangeNeighbors; y++)
                    {
                        if (!(x == 0 && y == 0))
                        {
                            ivec2 neighborPos = ivec2(agent.position.x + x, agent.position.y + y); 
                            neighborPos = ivec2(round(neighborPos));
                            vec4 neighborInfo = texelFetch(sTD2DInputs[INOUT_DRAW_MASS_COORD_BUFFER], ivec2(neighborPos), 0);
                            ivec2 neighborCoord = ivec2(neighborInfo.gb);
                            float neighborColor = neighborInfo.a;

                            if(neighborColor > 0 && neighborColor == agent.color.r)
                            {
                                float neighborHealth = texelFetch(sTD2DInputs[INOUT_DIRECTION_AND_HEALTH_BUFFER], ivec2(neighborCoord), 0).a;

                                if(neighborHealth >= 4)
                                {
                                    validMateFound = true;
                                    mateCoord = neighborCoord;

                                    float born = clamp(random2((sin(uTime * .849))), 0.0, 1.0);

                                    if(born < uReproductionChance && validMateFound == true)
                                    {
                                        ivec2 emptyCell = findEmptyCell(coord);

                                        Agent child;

                                        if (emptyCell != ivec2(-99))
                                        {
                                            child.myCoord = emptyCell;
                                            child = crossoverAndMutate(child.myCoord, agent, mateCoord);
                                            UpdateAgent(child);
                                            Write(child, foods, child.myCoord, -1); 
                                            WriteTrailsTexture(child, child.myCoord);
                                        }
                                    }

                                    else if (born < uReproductionChance && validMateFound == false && agent.health >= 6)
                                    {
                                        ivec2 emptyCell = findEmptyCell(coord);
                                        Agent child;

                                        if (emptyCell != ivec2(-99))
                                        {
                                            child.myCoord = emptyCell;
                                            child = crossoverAndMutate(child.myCoord, agent, mateCoord);
                                            UpdateAgent(child);
                                            Write(child, foods, child.myCoord, -1); 
                                            WriteTrailsTexture(child, child.myCoord);
                                        }
                                    }                                
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
            agent = GenerateEmptyAgent();
        }
    }
    else
    {
        agent = GenerateEmptyAgent();
    }

    Write(agent, foods, coord, closest);
    WriteTrailsTexture(agent, coord);
}


