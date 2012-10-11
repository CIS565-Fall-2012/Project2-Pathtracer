// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#define USE_FRESNEL_OR_SNELL	//Comment for SNELL, Uncomment for Fresnel

#include "intersections.h"
#include <time.h>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"

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
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection, float reflectanceCoeffient);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__  bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  return false;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
	//Incident ray is from Origin of Ray to Surface.
	float eta = incidentIOR / transmittedIOR;
	float dotValue = glm::dot(normal, -incident);	//Since we can a "V" shaped dot product // cos Theta1
	float k = 1.0f - (eta * eta * (1.0 - dotValue * dotValue));	//cos^2 Theta2
	if(k > 1.0f)
		return glm::vec3(0.0f, 0.0f, 0.0f);
	else
		return glm::normalize(eta * incident + (eta * dotValue - sqrt(k)) * normal);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  //nothing fancy here
	//return glm::normalize(glm::reflect(incident, normal));
	return glm::normalize(incident - (normal * (float)glm::dot(normal, incident) * 2.0f));
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
//__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection, float reflectanceCoeffient) {
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, float reflectanceCoeffient) 
{
	Fresnel fresnel;

	float Theta = acos(glm::dot(normal, -incident));
	//Do Schlick's Approximation of Fresnel Reflectance Coefficient
	fresnel.reflectionCoefficient =  reflectanceCoeffient + ((1 - reflectanceCoeffient) * pow((1.0f - cos(Theta)), 5.0f));
	fresnel.transmissionCoefficient = 1 - fresnel.reflectionCoefficient;
  
	//reflectionDirection = calculateReflectionDirection(normal, incident);
	//transmissionDirection = calculateTransmissionDirection(normal, incident, incidentIOR, transmittedIOR);
  
	//fresnel.reflectionCoefficient = 1;
	//fresnel.transmissionCoefficient = 0;
	return fresnel;
}

//LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2, int t) {
	
	//crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
	
	float up = sqrt(xi1); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = xi2 * TWO_PI;
	
	//Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
	normal = glm::normalize(normal);
	glm::vec3 directionNotNormal;
	
	if (abs(normal.x) <= SQRT_OF_ONE_THIRD && abs(normal.y) <= SQRT_OF_ONE_THIRD && abs(normal.z) <= SQRT_OF_ONE_THIRD) 
	{ 
		if(t%3 == 0)
			directionNotNormal = glm::vec3(1, 0, 0);
		else if(t%3 == 1)
			directionNotNormal = glm::vec3(0, 1, 0);
		else
			directionNotNormal = glm::vec3(0, 0, 1);
	}
	else if (abs(normal.x) <= SQRT_OF_ONE_THIRD && abs(normal.y) <= SQRT_OF_ONE_THIRD) 
	{ 
		if(t%2 == 0)
			directionNotNormal = glm::vec3(1, 0, 0);
		else
			directionNotNormal = glm::vec3(0, 1, 0);
	} 
	else if (abs(normal.y) <= SQRT_OF_ONE_THIRD && abs(normal.z) <= SQRT_OF_ONE_THIRD) 
	{ 
		if(t%2 == 0)
			directionNotNormal = glm::vec3(0, 1, 0);
		else
			directionNotNormal = glm::vec3(0, 0, 1);
	} 
	else if (abs(normal.z) <= SQRT_OF_ONE_THIRD && abs(normal.x) <= SQRT_OF_ONE_THIRD) 
	{
		if(t%2 == 0)
			directionNotNormal = glm::vec3(0, 0, 1);
		else
			directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.x) <= SQRT_OF_ONE_THIRD) 
	{ 
		directionNotNormal = glm::vec3(1, 0, 0);
	} 
	else if (abs(normal.y) <= SQRT_OF_ONE_THIRD) 
	{ 
		directionNotNormal = glm::vec3(0, 1, 0);
	} 
	else if (abs(normal.z) <= SQRT_OF_ONE_THIRD) 
	{
		directionNotNormal = glm::vec3(0, 0, 1);
	}
	
	//Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1)); 
	
	return glm::normalize(( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 ));
	
}

//TODO: IMPLEMENT THIS FUNCTION
//Now that you know how cosine weighted direction generation works, try implementing non-cosine (uniform) weighted random direction generation. 
//This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {
	
	//float U = xi1, V = xi2;

	float Theta = TWO_PI * xi1;
	float Phi = asin(sqrt(xi2));
	
	float x = cos(Theta) * sin(Phi);
	float y = sin(Theta) * sin(Phi);
	float z = cos(Phi);
  
	return glm::normalize(glm::vec3(x,y,z));
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, staticGeom g, int closestIndex, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& color, glm::vec3& unabsorbedColor, material m, material* mats, float xi1, float xi2, float xi3, int t)
{
	float ReflectivityFactor = m.hasReflective;
	float RefractivityFactor = m.hasRefractive;
	float DiffuseFactor;
	
	if(RefractivityFactor == 0.0 && ReflectivityFactor == 0.0)
	{
		DiffuseFactor = 1.0f;
	}
	else if(RefractivityFactor == 0.0 && ReflectivityFactor != 0.0)
	{
		 DiffuseFactor= 1.0f - ReflectivityFactor;
	}
	else if(ReflectivityFactor == 0.0 && RefractivityFactor != 0.0)
	{
		DiffuseFactor = 1.0f - RefractivityFactor;
	}
	else if(m.color == glm::vec3(0.0, 0.0, 0.0))
	{
		ReflectivityFactor = ReflectivityFactor / (ReflectivityFactor + RefractivityFactor);
		RefractivityFactor = RefractivityFactor / (ReflectivityFactor + RefractivityFactor);
		DiffuseFactor = 0;
	}
	else
	{
		DiffuseFactor = 2.0 - ReflectivityFactor - RefractivityFactor;
		ReflectivityFactor /= 2.0f;
		RefractivityFactor /= 2.0f;
		DiffuseFactor /= 2.0f;
	}

	int type = -1;
	if(xi3 >= 0.0 && xi3 <= ReflectivityFactor)
		type = 1;
	else if(xi3 > ReflectivityFactor && xi3 <= ReflectivityFactor + RefractivityFactor)
		type = 2;
	else
		type = 0;
#ifdef USE_FRESNEL_OR_SNELL
	//Fresnel Refraction Code
	if(type == 2)//if(m.hasRefractive > 0.0f)
	{
		float AirIOR = 1.0f;
		/*Checking whether the Intersection is inside the same object but with a different object that is intersecting the current object
		//if(closestIndex != r.currentObjIndex && r.isInside)	
		//{
		//	glm::vec3 tempDir = calculateTransmissionDirection(normal, r.direction, 
		//								mats[r.currentMatIndex].indexOfRefraction , m.indexOfRefraction);
		//	if(tempDir == glm::vec3(0.0, 0.0, 0.0)) //Internal Reflection
		//	{
		//		r.direction = calculateReflectionDirection(normal, r.direction);
		//		r.origin = intersect + r.direction * 0.002f;
		//		r.keep = 1;
		//		color = m.color;
		//		r.isInside = true;	//ray is exiting an object
		//		//r.currentObjIndex - wont change
		//		//r.currentMatIndex - wont change
		//	}
		//	else	//Refraction
		//	{
		//		r.direction = glm::normalize(tempDir);
		//		r.origin = intersect + r.direction * 0.002f;
		//		r.keep = 1;
		//		color = m.color;
		//		r.isInside = true;	//ray is exiting an object
		//		r.currentObjIndex = closestIndex;	//storing the index of air
		//		r.currentMatIndex = g.materialid;
		//	}
		//}*/
		//Checking whether the intersection is with the same object the ray is within
		if(r.isInside && closestIndex == r.currentObjIndex)
		{
			Fresnel F = calculateFresnel(normal, r.direction, m.indexOfRefraction, AirIOR, m.hasReflective);
			if(xi1 <= F.reflectionCoefficient)	//Reflection xi1 >= 0 && 
			{
				r.direction = calculateReflectionDirection(normal, r.direction);
				r.origin = intersect + r.direction * 0.005f;
				r.keep = 1;
				color = m.color;
				r.isInside = true;//does not change
				//r.currentObjIndex does not change
				//r.currentMatIndex does not change

				return 1;
			}
			else if(xi1 > F.reflectionCoefficient)// && xi1 <= 1)	//Transmission
			{
				//Ray is going from inside the object to Air
				glm::vec3 tempDir = calculateTransmissionDirection(normal, r.direction, m.indexOfRefraction, AirIOR);
				if(tempDir == glm::vec3(0.0, 0.0, 0.0)) //Internal Reflection
				{
					r.direction = calculateReflectionDirection(normal, r.direction);
					r.origin = intersect + r.direction * 0.002f;
					r.keep = 1;
					color = m.color;
					r.isInside = true;	//ray is exiting an object
					//r.currentObjIndex = -1;	//storing the index of air
					//r.currentMatIndex = -1;

					return 1;	
				}
				else	//Refraction
				{
					r.direction = glm::normalize(tempDir);
					r.origin = intersect + r.direction * 0.002f;
					r.keep = 1;
					color = m.color;
					r.isInside = false;	//ray is exiting an object
					r.currentObjIndex = -1;	//storing the index of air
					r.currentMatIndex = -1;

					return 2;
				}
			}
		}
		//Checking whether the intersection is from outside, ie. from Air to object
		else if(!r.isInside)
		{
			Fresnel F = calculateFresnel(normal, r.direction, AirIOR, m.indexOfRefraction, m.hasReflective);
			if(xi1 <= F.reflectionCoefficient)	//Reflection xi1 >= 0 && 
			{
				r.direction = calculateReflectionDirection(normal, r.direction);
				r.origin = intersect + r.direction * 0.005f;
				r.keep = 1;
				color = m.color;
				r.isInside = false; //does not change
				//r.currentObjIndex does not change
				//r.currentMatIndex does not change

				return 1;
			}
			else if(xi1 > F.reflectionCoefficient)// && xi1 <= 1)	//Transmission
			{
				glm::vec3 tempDir = calculateTransmissionDirection(normal, r.direction, AirIOR, m.indexOfRefraction);
				if(tempDir == glm::vec3(0.0, 0.0, 0.0)) //Reflection
				{
					r.direction = calculateReflectionDirection(normal, r.direction);
					r.origin = intersect + r.direction * 0.002f;
					r.keep = 1;
					color = m.color;
					r.isInside = false;	//ray is exiting an object
					r.currentObjIndex = -1;	//storing the index of air
					r.currentMatIndex = -1;	

					return 1;
				}
				else
				{
					r.direction = glm::normalize(tempDir);
					r.origin = intersect + r.direction * 0.002f;
					r.keep = 1;
					color = m.color;
					r.isInside = true;	//ray is exiting an object
					r.currentObjIndex = closestIndex;	//storing the index of air
					r.currentMatIndex = g.materialid;

					return 2;
				}
			}
		}
	}
#else
	//SNELL REFRACTION CODE
	if(type == 2)//if(m.hasRefractive > 0.0f)
	{
		float AirIOR = 1.0f;
		//Checking whether the Intersection is inside the same object but with a different object that is intersecting the current object
		//if(closestIndex != r.currentObjIndex && r.isInside)	
		//{
		//	glm::vec3 tempDir = calculateTransmissionDirection(normal, r.direction, 
		//								mats[r.currentMatIndex].indexOfRefraction , m.indexOfRefraction);
		//	if(tempDir == glm::vec3(0.0, 0.0, 0.0)) //Internal Reflection
		//	{
		//		r.direction = calculateReflectionDirection(normal, r.direction);
		//		r.origin = intersect + r.direction * 0.002f;
		//		r.keep = 1;
		//		color = m.color;
		//		r.isInside = true;	//ray is exiting an object
		//		//r.currentObjIndex - wont change
		//		//r.currentMatIndex - wont change
		//	}
		//	else	//Refraction
		//	{
		//		r.direction = glm::normalize(tempDir);
		//		r.origin = intersect + r.direction * 0.002f;
		//		r.keep = 1;
		//		color = m.color;
		//		r.isInside = true;	//ray is exiting an object
		//		r.currentObjIndex = closestIndex;	//storing the index of air
		//		r.currentMatIndex = g.materialid;
		//	}
		//}
		//Checking whether the intersection is with the same object the ray is within
		if(r.isInside && closestIndex == r.currentObjIndex)
		{
			glm::vec3 tempDir = calculateTransmissionDirection(normal, r.direction, m.indexOfRefraction, AirIOR);
			if(tempDir == glm::vec3(0.0, 0.0, 0.0)) //Internal Reflection
			{
				r.direction = calculateReflectionDirection(normal, r.direction);
				r.origin = intersect + r.direction * 0.002f;
				r.keep = 1;
				color = m.color;
				r.isInside = true;	//ray is exiting an object
				//r.currentObjIndex = -1;	//storing the index of air
				//r.currentMatIndex = -1;

				return 1;
			}
			else	//Refraction
			{
				r.direction = glm::normalize(tempDir);
				r.origin = intersect + r.direction * 0.002f;
				r.keep = 1;
				color = m.color;
				r.isInside = false;	//ray is exiting an object
				r.currentObjIndex = -1;	//storing the index of air
				r.currentMatIndex = -1;

				return 2;
			}
		}
		//Checking whether the intersection is from outside, ie. from Air to object
		else
		{
			glm::vec3 tempDir = calculateTransmissionDirection(normal, r.direction, AirIOR, m.indexOfRefraction);
			if(tempDir == glm::vec3(0.0, 0.0, 0.0)) //Internal Reflection
			{
				r.direction = calculateReflectionDirection(normal, r.direction);
				r.origin = intersect + r.direction * 0.002f;
				r.keep = 1;
				color = m.color;
				r.isInside = false;	//ray is exiting an object
				//r.currentObjIndex = -1;	//storing the index of air
				//r.currentMatIndex = -1;

				return 1;
			}
			else
			{
				r.direction = tempDir;
				r.origin = intersect + r.direction * 0.002f;
				r.keep = 1;
				color = m.color;
				r.isInside = true;	//ray is exiting an object
				r.currentObjIndex = closestIndex;	//storing the index of air
				r.currentMatIndex = g.materialid;

				return 2;
			}
		}

		return 2;
	}
	
#endif
	else if(type == 1)//else if(m.hasReflective > 0.0f + 0.001f)
	{
		r.direction = calculateReflectionDirection(normal, r.direction);
		r.origin = intersect + r.direction * 0.005f;
		r.keep = 1;
		color = m.color;

		return 1;
	}
	else if(type == 0)
	{
		if(RefractivityFactor == 0.0f && ReflectivityFactor < 1.0f)
		{
			r.direction = calculateRandomDirectionInHemisphere(glm::normalize(normal), xi1, xi2, t);
			r.origin = intersect + r.direction * 0.002f;
			r.keep = 1;
			color = m.color;

			return 0;
		}
		else if(RefractivityFactor < 1.0f && ReflectivityFactor == 0.0f)
		{
			r.direction = calculateRandomDirectionInHemisphere(glm::normalize(-normal), xi1, xi2, t);	//Generate Ray in refractive hemisphere
			r.origin = intersect + r.direction * 0.002f;
			r.keep = 1;
			color = m.color;

			return 0;
		}
		else
		{
			r.direction = getRandomDirectionInSphere(xi1, xi2);	//Generate Ray in refractive hemisphere
			r.origin = intersect + r.direction * 0.002f;
			r.keep = 1;
			color = m.color;

			return 0;
		}
	}
	return -1;
};

#endif
	