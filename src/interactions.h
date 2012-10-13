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
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

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
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
	float n = incidentIOR / transmittedIOR;
	float cosAngle = glm::dot(normal, -incident);  
	float k = 1.0f - ( n*n * (1.0 - cosAngle * cosAngle)); 
	glm::vec3 transmission = n * incident + (n * cosAngle - sqrt(k)) * normal;
		
	return glm::normalize(transmission);
	//return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) 
{
	//nothing fancy here, R=2(N.L)N+L
	//return glm::vec3(0,0,0);
	glm::vec3 rlect = incident - 2.0f * glm::dot(incident,normal) * normal ;
	return rlect;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
  Fresnel fresnel;
	//fresnel.reflectionCoefficient = 1;
	//fresnel.transmissionCoefficient = 0;

	float cosAngleIn = glm::dot(incident, normal);
	float n1 = incidentIOR, n2 = transmittedIOR;
	float tmpA = (n1*n1) / (n2*n2) * (1.0f - cosAngleIn * cosAngleIn);
	float cosAngleTran = sqrt(1.0f - tmpA);
	
	float Rs = ( (n1*cosAngleIn - n2*cosAngleTran) / ( n1*cosAngleIn + n2*cosAngleTran)) *
			   ( (n1*cosAngleIn - n2*cosAngleTran) / ( n1*cosAngleIn + n2*cosAngleTran)) ;
	float Rp = ( (n1*cosAngleTran - n2*cosAngleIn) / ( n2*cosAngleIn + n1*cosAngleTran)) *
			   ( (n1*cosAngleTran - n2*cosAngleIn) / ( n2*cosAngleIn + n1*cosAngleTran)) ;

	fresnel.reflectionCoefficient = (Rs + Rp) / 2.0f;
	fresnel.transmissionCoefficient = 1 - fresnel.reflectionCoefficient;

	return fresnel;
}

//LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
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
{//??????????what is xi1 and xi2???????????
	//assume they are random numbers in [0,1];
	float theta =  TWO_PI * xi1;
	float phi = acos( 2*xi2 - 1);

	return (  glm::vec3( cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi) )  );

}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor, 
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, 
                                      glm::vec3& color, glm::vec3& unabsorbedColor, material m, float xi1, float xi2,int closestGeomIndex){
	//Reflective material
	if(m.hasReflective > 0)
	{
		 r.continueFlag = true;
		 r.direction = calculateReflectionDirection(normal, r.direction);
		 r.origin = intersect + r.direction * 0.01f;
		 color = m.color;
	}
	
	//Refractive material--still working on it
	else if(m.hasRefractive > 0) ;
	
	//Diffuse material
	else
	{
		r.continueFlag = true;
		r.direction = calculateRandomDirectionInHemisphere(glm::normalize(normal), xi1, xi2);
		r.origin = intersect + r.direction * 0.01f;
		color = m.color;
	}
	
	return 0;
};

#endif
    