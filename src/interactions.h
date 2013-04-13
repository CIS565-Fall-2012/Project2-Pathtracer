// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"



struct Fresnel {
  float reflectionCoefficient;
  float transmissionCoefficient;
};

struct AbsorptionAndScatteringProperties{
    glm::vec3 absorptionCoefficient;
    float reducedScatteringCoefficient; 
};

//forward declaration
__host__ __device__  bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
	glm::vec3 transmitted;
	transmitted.x = pow((float)E, (float)(-1 * absorptionCoefficient.x * distance));
	transmitted.y = pow((float)E, (float)(-1 * absorptionCoefficient.y * distance));
	transmitted.z = pow((float)E, (float)(-1 * absorptionCoefficient.z * distance));
	return transmitted;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__  bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, 
                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  
	
															
	return false;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
  
  float n12 = incidentIOR/transmittedIOR;

		float cos1 = glm::dot(normal, -1.0f*incident);
		float rootValue = 1 - (n12*n12)*(1.0f-cos1*cos1);
		if (rootValue < 0)
		{
			// Internal Reflection
			//internalReflection = true;
			return calculateReflectionDirection(normal, incident);
		}

		if (cos1 > 0.0)
			return glm::normalize(normal*(n12*cos1 - sqrt(rootValue)) + incident*n12);
		else
			return glm::normalize(normal*(-n12*cos1 + sqrt(rootValue)) + incident*n12);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  //nothing fancy here

	normal=glm::normalize(normal);
	incident=glm::normalize(incident);
	glm::vec3 reflectedRay;
	reflectedRay= incident - (normal +normal)*(glm::dot(incident,normal));
	reflectedRay = glm::normalize(reflectedRay);

	return reflectedRay;
  
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR)
{
  Fresnel fresnel;

	incident = glm::normalize(incident);
	normal = glm::normalize(normal);
	float cosIncidence = abs(glm::dot(incident, normal));
	float sinIncidence = sqrt(1-cosIncidence*cosIncidence);

	if (transmittedIOR > 0.0 && incidentIOR > 0)
	{
		float sinRefract = (incidentIOR/transmittedIOR)*sinIncidence;
		float commonNumerator = sqrt(1-sinRefract*sinRefract);
		float RsNumerator = incidentIOR*cosIncidence-transmittedIOR*commonNumerator;
		float RsDenominator = incidentIOR*cosIncidence+transmittedIOR*commonNumerator;
		float Rs = 0;
		if (RsDenominator != 0)
			Rs = (RsNumerator/RsDenominator)*(RsNumerator/RsDenominator);

		float RpNumerator = (incidentIOR * commonNumerator) - (transmittedIOR * cosIncidence);
		float RpDenominator = (incidentIOR * commonNumerator) + (transmittedIOR * cosIncidence);
		float Rp = 0;
		if (RpDenominator != 0)
			Rp = (RpNumerator/RpDenominator)*(RpNumerator/RpDenominator);
		
		fresnel.reflectionCoefficient = (Rs + Rp)/2.0;
		fresnel.transmissionCoefficient = 1 - fresnel.reflectionCoefficient;
	}
	else
	{
		fresnel.reflectionCoefficient = 1;
		fresnel.transmissionCoefficient = 0;
	}
	return fresnel;
}

//LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
    //crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
	float up = (sqrt(xi1)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta) 
    float around = xi2 * TWO_PI;
    
    //Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) { 
      directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) { 
      directionNotNormal = glm::vec3(0, 1, 0);
    } else {
      directionNotNormal = glm::vec3(0, 0, 1);
    }
    
    //Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 = (glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = (glm::cross(normal, perpendicularDirection1)); 
    
    return glm::normalize( ( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 ));
    
}

//TODO: IMPLEMENT THIS FUNCTION
//Now that you know how cosine weighted direction generation works, try implementing non-cosine (uniform) weighted random direction generation. 
//This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {



	float up = xi1 * 2 - 1; // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;

	return glm::vec3( up, cos(around) * over, sin(around) * over );

  
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
//__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor, 
//                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, 
//                                       glm::vec3& color, glm::vec3& unabsorbedColor, material m){

__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3& color, material m,glm::vec3 random_direction,float bounces,
										glm::vec2 resolution, int index)
{


	glm::vec3 temp_color;
	//glm::vec3 random_direction;
	if(m.hasReflective==0 && m.hasRefractive==0)
	{
			
			r.direction=(random_direction);
			//r.direction=random_direction;
						
			r.origin=intersect;
			if (bounces==1)
			{
				temp_color=m.color;
				
			}
			else
			{
				temp_color*=m.color;
			}
			color=temp_color;
	}
	return 1;
 
};

#endif
    