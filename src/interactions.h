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
__host__ __device__ glm::vec3 calculateTransmissionDirection(const glm::vec3 &normal, const glm::vec3 &incidentRayDirection, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(const glm::vec3 &normal, const glm::vec3 &incidentRayDirection);
__host__ __device__ Fresnel calculateFresnel(const glm::vec3 &normal, const glm::vec3 &incidentRayDirection, float incidentIOR, float transmittedIOR, const glm::vec3 &transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(const glm::vec3 &normal, float xi1, float xi2);

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__  bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, 
                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  return false;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmissionDirection(const glm::vec3 &normal, const glm::vec3 &incident, float incidentIOR, float transmittedIOR) {
	float cosTheta = glm::dot(-incident, normal);
	float ratio = incidentIOR / transmittedIOR;
	float sinSqrTheta = 1.0f - ratio * ratio * (1 - cosTheta * cosTheta);
	if(sinSqrTheta > 1)
	{
		return glm::vec3(-10000, -10000, -10000);
	}
	else
	{
		return (ratio * incident + (ratio * cosTheta - sqrt(sinSqrTheta)) * normal);
	}
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(const glm::vec3 &normal, const glm::vec3 &incident) {
  //nothing fancy here
	return (incident - 2.0f * glm::dot(incident, normal) * normal);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
//NOT Using Schlick's approximation
__host__ __device__ Fresnel calculateFresnel(const glm::vec3 &normal, const glm::vec3 &incident, float incidentIOR, float transmittedIOR, const glm::vec3 &transmissionDirection)
{
  Fresnel fresnel;
  float cosThetaI = glm::dot(-incident, normal);
  float sinSqrThetaT = (incidentIOR / transmittedIOR) * (incidentIOR / transmittedIOR) * (1.0f - cosThetaI * cosThetaI);
  float cosThetaT = sqrt(1.0f - sinSqrThetaT);
  float RPerp = ((incidentIOR * cosThetaI - transmittedIOR * cosThetaT) / (incidentIOR * cosThetaI + transmittedIOR * cosThetaT)) *
						((incidentIOR * cosThetaI - transmittedIOR * cosThetaT) / (incidentIOR * cosThetaI + transmittedIOR * cosThetaT));
  float RPara = ((transmittedIOR * cosThetaI - incidentIOR * cosThetaT) / (transmittedIOR * cosThetaI + incidentIOR * cosThetaT)) *
						((transmittedIOR * cosThetaI - incidentIOR * cosThetaT) / (transmittedIOR * cosThetaI + incidentIOR * cosThetaT));

  fresnel.reflectionCoefficient = (RPerp + RPara) / 2.0f;
  fresnel.transmissionCoefficient = 1.0f - fresnel.reflectionCoefficient;
  return fresnel;
}

//LOOK: This function demonstrates cosine weighted random direction generation in a hemisphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(const glm::vec3 &normal, float xi1, float xi2) {
    
    //crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
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
    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1)); 
    
    return ( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 );
    
}

//TODO: IMPLEMENT THIS FUNCTION
//Now that you know how cosine weighted direction generation works, try implementing non-cosine (uniform) weighted random direction generation. 
//This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2)
{
	//float x = 2.0f * cos(TWO_PI * xi1) * sqrt(xi2 * (1.0f - xi2));
	//float y = 2.0f * sin(TWO_PI * xi1) * sqrt(xi2 * (1.0f - xi2));
	//float z = 1.0f - 2.0f * xi2;
	return glm::normalize(glm::vec3((2.0f * cos(TWO_PI * xi1) * sqrt(xi2 * (1.0f - xi2)), 2.0f * sin(TWO_PI * xi1) * sqrt(xi2 * (1.0f - xi2)), 1.0f - 2.0f * xi2)));
}



#endif
    