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
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, bool& internalReflection);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

void GetParametersForRayCast(glm::vec2 resolution, glm::vec3 eye, glm::vec3 view, glm::vec3 up, 
	                        glm::vec2 fov, glm::vec3& M, glm::vec3& A, glm::vec3& B, float& distImagePlaneFromCamera){
  ray r;
  r.origin = eye;
  view = glm::normalize(view);
  up = glm::normalize(up);

  A = glm::normalize(glm::cross(view, up));
  B = glm::normalize(glm::cross(A, view));

  float tanVert = tan(fov.y*PI/180);
  float tanHor = tan(fov.x*PI/180);

  float camDistFromScreen = (float)((resolution.y/2.0)/tanVert);
  glm::vec3 C = view*camDistFromScreen;
  M = eye + C;

  distImagePlaneFromCamera = camDistFromScreen;
}

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
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, bool& internalReflection) {
		float n12 = incidentIOR/transmittedIOR;

		float cos1 = glm::dot(normal, glm::vec3(-incident.x, -incident.y, -incident.z));
		float rootValue = 1 - (n12*n12)*(1.0f-cos1*cos1);
		if (rootValue < 0)
		{
			// Internal Reflection
			internalReflection = true;
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
  	// Rr = Ri - 2N(Ri.N)
	float dotProd = glm::dot(incident, normal);
	return glm::normalize(incident - normal*2.0f*dotProd);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
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
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    // Find a direction that is not the normal based off of whether or not the normal's components 
	// are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    glm::vec3 directionNotNormal = glm::vec3(0,0,0);
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) { 
      directionNotNormal.x = 1;
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) { 
      directionNotNormal.y = 1;
    } else {
      directionNotNormal.z = 1;
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
	float z = 1.0f - 2.0f*xi1;
	float temp = 1.0f - z*z;
    float r;
	if (temp < 0.0f)
		r = 0.0f;
	else
		r = sqrtf(temp);

    float phi = 2.0f * PI * xi2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return glm::normalize(glm::vec3(x, y, z));
}

__host__ __device__ void CookTorrence(ray inRay, ray& outRay, glm::vec3 intersect, glm::vec3 N, glm::vec3 emittedColor, 
	                                  glm::vec3& color, material mat, bool inside, float xi1, float xi2)
{
	// Reference: http://renderman.pixar.com/view/cook-torrance-shader
	float gaussConstant = 100;
	float m = 0.2; // roughness

	// Calculate reflective and transmittive coefficients using Fresnel equations
	float n1, n2;
	if (inside)
	{
		n2 = 1.0f;
		n1 = mat.indexOfRefraction;
	}
	else
	{
		n1 = 1.0f;
		n2 = mat.indexOfRefraction;
	}

	Fresnel fresnel = calculateFresnel(N, intersect, n1, n2);
	outRay.direction = calculateRandomDirectionInHemisphere(N, xi1, xi2);
	glm::vec3 L = outRay.direction;

	glm::vec3 V = glm::normalize(inRay.direction * -1.0f);
	
	// Half angle vector
	glm::vec3 H = glm::normalize(V+L);

	// Attenuation Factor G
	float NDotV = glm::dot(N, V);
	float NDotH = glm::dot(N, H);
	float VDotH = glm::dot(V, H);
	float NDotL = glm::dot(N, L);

	float G = glm::min(2*NDotH*NDotV/VDotH, 2*NDotH*NDotL/VDotH);
	G = glm::min(1.0f, G);

	// Microfacet Slope Distribution D
	float alpha = acos(NDotH);
	float D = gaussConstant*glm::exp(-(alpha*alpha)/(m*m));
	float cook = (fresnel.reflectionCoefficient*D*G)/(PI*NDotL*NDotV);
	color = emittedColor * mat.color;// * cook;
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray inRay, ray& outRay, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor, 
                                       /*AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, */
                                       glm::vec3& color, /*glm::vec3& unabsorbedColor,*/ material m, bool inside, float xi1, float xi2, float xi3){
	outRay.origin = intersect;

	if (m.reflectivity >= (1.0f-0.01f))
	{
		// Specular reflection
		outRay.direction = calculateReflectionDirection(normal, inRay.direction);
		color = emittedColor;
		return 1;
	}
	else if (m.hasRefractive)
	{
		// Calculate reflective and transmittive coefficients using Fresnel equations
		float n1, n2;
		if (inside)
		{
			n2 = 1.0f;
			n1 = m.indexOfRefraction;

			// Do refraction
			bool internalReflection = false;
			outRay.direction = calculateTransmissionDirection(normal, inRay.direction, n1, n2, internalReflection);
			color = emittedColor;//*fresnel.transmissionCoefficient;
			if (internalReflection)
				return 1;
			else
				return 2;
		}
		else
		{
			n1 = 1.0f;
			n2 = m.indexOfRefraction;
		}

		Fresnel fresnel = calculateFresnel(normal, inRay.direction, n1, n2);
		
		// Use Russian roulette to determine whether to reflect or refract based on the refractive index
		float russianRoulette = xi3;
		if (russianRoulette <= fresnel.transmissionCoefficient)
		{
			// Do refraction
			bool internalReflection = false;
			outRay.direction = calculateTransmissionDirection(normal, inRay.direction, n1, n2, internalReflection);
			color = emittedColor*fresnel.transmissionCoefficient;
			if (internalReflection)
				return 1;
			else
				return 2;
		}
		else
		{
			// Do reflection
			outRay.direction = calculateReflectionDirection(normal, inRay.direction);
			color = emittedColor*fresnel.reflectionCoefficient;
			return 1;
		}
	}
	else
	{
		//// Maybe partly specular
		//if (m.specularExponent > 0.0f)
		//{
		//	// Specular reflection
		//	outRay.direction = calculateReflectionDirection(normal, inRay.direction);
		//	color = emittedColor;
		//	return 1;
		//}
		//// Diffuse surface

		// If surface has reflectivity as well as is diffuse, then run russian rouleete to determine which should be run
		float russianRouletteGlossy = xi3;
		if (m.reflectivity > 0 && russianRouletteGlossy <= m.reflectivity)
		{
			// Reflect the ray
			outRay.direction = calculateReflectionDirection(normal, inRay.direction);
			color = emittedColor;
			return 1;
		}

		// Run diffuse if we reach here
		outRay.direction = calculateRandomDirectionInHemisphere(normal, xi1, xi2);
		color = emittedColor*m.color;
		//return 0;
		//CookTorrence(inRay, outRay, intersect, normal, emittedColor, color, m, inside, xi1, xi2);
		return 0;
	}
};

#endif
    