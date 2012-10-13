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
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 &reflectionDirection, glm::vec3 &transmissionDirection);
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
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  //nothing fancy here
  //return glm::vec3(0,0,0);
	return glm::normalize(incident-2.0f*normal*glm::dot(normal,incident)); 
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 &reflectionDirection, glm::vec3 &transmissionDirection) {
    Fresnel fresnel;
  float incidentAngle=acos(glm::dot(normal,-incident));
  
  if(incidentIOR>transmittedIOR){
		float CriticalAngle=asin(transmittedIOR/incidentIOR);
		if(incidentAngle>CriticalAngle){//pure reflection
		fresnel.reflectionCoefficient = 1.0f; 
		fresnel.transmissionCoefficient =0.0f;
		reflectionDirection=calculateReflectionDirection(normal,incident);
		return fresnel;
	   }
  }
 
	float TransmittedAngle=asin(sin(incidentAngle)*incidentIOR/transmittedIOR);
	float Rs=max(pow((incidentIOR*cos(incidentAngle)-transmittedIOR*cos(TransmittedAngle))/(incidentIOR*cos(incidentAngle)+transmittedIOR*cos(TransmittedAngle)),2.0f),0.0f);
	float Rt=max(pow((incidentIOR*cos(TransmittedAngle)-transmittedIOR*cos(incidentAngle))/(incidentIOR*cos(TransmittedAngle)+transmittedIOR*cos(incidentAngle)),2.0f),0.0f);
	fresnel.reflectionCoefficient =(Rs+Rt)/2;
	fresnel.transmissionCoefficient = 1-fresnel.reflectionCoefficient;
	reflectionDirection=calculateReflectionDirection(normal,incident);
	/*glm::vec3 crs=glm::normalize(glm::cross(incident,normal));
	glm::vec3 axs=glm::normalize(glm::cross(normal,crs));
	transmissionDirection=glm::normalize(tan(TransmittedAngle)*axs-normal);*/
	float n12=incidentIOR/transmittedIOR;
	float ni=glm::dot(normal,incident);
	float squareValue=1.0f-n12*n12*(1.0f-ni*ni);
	transmissionDirection=glm::normalize((-n12*ni-sqrt(squareValue))*normal+n12*incident);

  return fresnel;
}

//LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
   // crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
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
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {
   
  float x=TWO_PI*cos(TWO_PI*xi1)*sqrt(xi2*(1-xi2));
  float y=TWO_PI*sin(TWO_PI*xi1)*sqrt(xi2*(1-xi2));
  float z=PI*(1-xi2);
  return glm::vec3(x,y,z);
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(glm::vec3 eye,ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor, 
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, 
                                       glm::vec3& color, glm::vec3& unabsorbedColor, material m,float time, rayData &newrayData){

   thrust::default_random_engine rng(hash(time));
    thrust::uniform_real_distribution<float> u01(0.0f,1.0f);
 
	glm::vec3 reflectedDirection,transmittedDirection;
	Fresnel fr;

	//difffuse surface
  if((!m.hasReflective)&&(!m.hasRefractive))
  {
	  newrayData.newray.origin=intersect+0.001f*normal;
	  newrayData.newray.direction=glm::normalize(calculateRandomDirectionInHemisphere(normal,u01(rng),u01(rng)));
	  color=m.color;
	  if(m.specularExponent>0.0f){
		  glm::vec3 viewDir=glm::normalize(eye-intersect);
		  glm::vec3 lightReflection=calculateReflectionDirection(normal,-newrayData.newray.direction);
		  float specularTerm=pow(max(glm::dot(viewDir,lightReflection),0.0f),m.specularExponent);
		  color+=specularTerm*m.specularColor;
	  }

	 // if(m.indexOfRefraction>0.0f){
	//   fr=calculateFresnel(normal,r.direction,1.0f,m.indexOfRefraction,reflectedDirection,transmittedDirection);	
	//glm::vec3 viewDirection=glm::normalize(eye-intersect);
	//glm::vec3 HalfAngleVector=glm::normalize(newrayData.newray.direction+glm::normalize(viewDirection));
	//
	//float vn=(glm::dot(viewDirection,normal));
	//float hn=(glm::dot(HalfAngleVector,normal));
	//float lh=(glm::dot(newrayData.newray.direction,HalfAngleVector));
	//float vh=(glm::dot(viewDirection,HalfAngleVector));
	//float ln=(glm::dot(newrayData.newray.direction,normal));
	//	//Beckmann distribution
	//float roughness=0.05f;
	//float tan2alpha=(1.0f/hn)/hn-1.0f;
	//float D=exp(-tan2alpha/roughness/roughness)/(roughness*hn*roughness*hn*PI*hn*hn);
 //   //GAUSSIAN;
	////  float alpha = acos(hn);
	////D=exp(-(alpha/roughness/roughness);
	////float F=fr.reflectionCoefficient;
	////schlick's approximation
	////fr=calculateFresnel(normal,normal,1.0f,m.indexOfRefraction,reflectedDirection,transmittedDirection);	
	////float n1=m.indexOfRefraction;
	////float rani=pow((float)(1.0f-n1)/(1.0f+n1),2.0f);
	////float F = rani+(1-rani)*pow( 1.0f - vh, 5.0f );
	//float F=fr.reflectionCoefficient;
	//float G=min(1.0f,2*hn/vh*min(vn,ln));
	//float specularTerm= max(D/4.0*F/vn*G/ln,0.0);
	//color+=specularTerm*m.specularColor;
	//  }
	  return 0;
  }    
		if(glm::dot(normal,r.direction)<0.001)
		fr=calculateFresnel(normal,r.direction,1.0f,m.indexOfRefraction,reflectedDirection,transmittedDirection);	
		else 
		fr=calculateFresnel(-normal,r.direction,m.indexOfRefraction,1.0f,reflectedDirection,transmittedDirection);	
 
  bool reflectiveRay=false,refractiveRay=false;
  if((m.hasReflective)&&(m.hasRefractive)){
	  if(abs(fr.transmissionCoefficient)<0.1){
		  reflectiveRay=true;
	  }else{
		 float russianRoulette = (float)u01(rng);
		if(russianRoulette<0.5)
		 reflectiveRay=true;
		else
		 refractiveRay=true;
	  }
	 }else if(m.hasReflective){
		 reflectiveRay=true;
	 }else if(m.hasRefractive){
		 refractiveRay=true;
	 }


  if(reflectiveRay){
	  newrayData.newray.origin=intersect+0.001f*normal;
	  newrayData.newray.direction=reflectedDirection;
	  newrayData.coeff=fr.reflectionCoefficient;
	  color=m.color*fr.reflectionCoefficient;
	    if(m.indexOfRefraction>0.0f){
		  glm::vec3 viewDir=glm::normalize(eye-intersect);
		  glm::vec3 lightReflection=calculateReflectionDirection(normal,-newrayData.newray.direction);
		  float specularTerm=pow(max(glm::dot(viewDir,lightReflection),0.0f),m.specularExponent);
		  color+=specularTerm*m.specularColor;
	  }
	  return 1;
  }
  if(refractiveRay){
	  newrayData.newray.origin=intersect+0.001f*normal;
	  newrayData.newray.direction=transmittedDirection;
	  newrayData.coeff=fr.transmissionCoefficient;
	  color=m.color*fr.transmissionCoefficient;
    return 2;
  }

  return 1;
};

#endif
    