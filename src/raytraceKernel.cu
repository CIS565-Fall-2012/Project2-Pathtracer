// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <thrust\remove.h>
#include <thrust\count.h>
#include <thrust\device_vector.h>
#include <thrust\device_ptr.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>


//#define DEPTH_OF_FIELD
// DO NOT set traceDepth to 1, else program would crash on account of x % 0 operation
#define traceDepth 10

__host__ __device__ int findNearestPrimitiveInRay(const staticGeom const *geoms, int numberOfGeoms, const ray &rt, glm::vec3 &intersectionPoint, glm::vec3 &normal);
__host__ __device__ int calculateBSDF(ray& rt, const glm::vec3 &intersectionPoint, const glm::vec3 &normal, glm::vec3 emittedColor, 
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, 
                                       glm::vec3& color, glm::vec3& unabsorbedColor, const material &m, const staticGeom const *geoms,
									   const int numberOfGeoms, const int geomIndex, const float randomNumber1, const float randomNumber2);

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);
  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(const glm::vec2 &resolution, float time, const int x, const int y, const glm::vec3 &eye, const glm::vec3 &View,
	const glm::vec3 &Up, const glm::vec2 &fov)
{
  ray r;
  r.origin = eye;
  
  // Distance of View Plane from eye
  float tanfovy = tan(fov.y);
  float dist = (resolution.y / 2.0f) / tanfovy;
  glm::vec3 view = glm::normalize(View);
  glm::vec3 up = glm::normalize(Up);
  glm::vec3 c = dist * view;
  glm::vec3 a = glm::cross(view, up);
  glm::vec3 b = glm::cross(a, view);

  //Center of screen
  glm::vec3 m = c + eye;

  //Using same vector a instead of a separate vector h
  a = (resolution.x / 2.0f) * a;
  
  //Using same vector b instead of a separate vector v
  b = (resolution.y / 2.0f) * b;

  //Anti-Aliasing
  thrust::default_random_engine rng(hash((x + y * resolution.x + 100.0f) * (time + 100.0f)));
  thrust::uniform_real_distribution<float> u05(-0.5f, 0.5f);
  float distortionInX = u05(rng);
  float distortionInY = u05(rng);

  //Point in space towards which ray has to be shot
  //glm::vec3 p = m + (2.0f * x / (resolution.x - 1.0f) - 1.0f) * a + (2.0f * y / (resolution.y - 1.0f) - 1.0f) * b;
  glm::vec3 p = m + (2.0f * ((float)x + distortionInX) / (resolution.x - 1.0f) - 1.0f) * a + (2.0f * ((float)y + distortionInY) / (resolution.y - 1.0f) - 1.0f) * b;
  r.direction = glm::normalize(p - eye);
  //r.direction = glm::normalize(r.direction);

  //Depth of Field
#ifdef DEPTH_OF_FIELD
  float focalLength = 12.0f;
  float apertureWidth = 0.75f;
  thrust::uniform_real_distribution<float> u001(-apertureWidth, apertureWidth);
  float cameraShake = u001(rng);

  glm::vec3 focalPlaneCenter = view * focalLength + eye;
  float lambda;
  if(fabs(r.direction.z) >= 0.001f)
  {
	lambda = (focalPlaneCenter.z - eye.z) / r.direction.z;
  }
  else if(fabs(r.direction.y) >= 0.001f)
  {
	lambda = (focalPlaneCenter.y - eye.y) / r.direction.y;
  }
  else if(fabs(r.direction.x) >= 0.001f)
  {
	lambda = (focalPlaneCenter.x - eye.x) / r.direction.x;
  }
  glm::vec3 intersectionPointOnFocalPlane = eye + lambda * r.direction;

  glm::vec3 newEyePos;
  if((int)time % 2 == 0)
  {
	  newEyePos = eye + a * (2.0f * cameraShake) / resolution.x;
  }
  else
  {
	  newEyePos = eye + b * (2.0f * cameraShake) / resolution.y;
  }

  r.direction = glm::normalize(intersectionPointOnFocalPlane - newEyePos);
  r.origin = newEyePos;
#endif

  return r;
}

__global__ void initialRaysGenerator(cameraData cam, ray *rays, glm::vec3* image, const int iterations)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(x <= cam.resolution.x && y <= cam.resolution.y)
	{
		int index = x + (y * cam.resolution.x);
		int temp = (cam.resolution.x - x) + ((cam.resolution.y - y) * cam.resolution.x);
		rays[index] = raycastFromCameraKernel(cam.resolution, (float)iterations, x, y, cam.position, cam.view, cam.up, cam.fov);
		rays[index].pixelIndex = temp;
		rays[index].survivalProbability = 1.0f;

		image[temp] = glm::vec3(1, 1, 1);
    }
	//__syncthreads();
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(const glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
	//__syncthreads();
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, const glm::vec2 resolution, glm::vec3* image, glm::vec3 *currentFrame, int iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
	  image[index].x = ((float)image[index].x * (float)(iterations - 1) + currentFrame[index].x) / iterations;
	  image[index].y = ((float)image[index].y * (float)(iterations - 1) + currentFrame[index].y) / iterations;
	  image[index].z = ((float)image[index].z * (float)(iterations - 1) + currentFrame[index].z) / iterations;

	  color.x = image[index].x * 255.0f;
      color.y = image[index].y * 255.0f;
      color.z = image[index].z * 255.0f;

      if(color.x>255.0f){
        color.x = 255.0f;
      }

      if(color.y>255.0f){
        color.y = 255.0f;
      }

      if(color.z>255.0f){
        color.z = 255.0f;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = (unsigned char)0;
      PBOpos[index].x = (unsigned char)color.x;     
      PBOpos[index].y = (unsigned char)color.y;
      PBOpos[index].z = (unsigned char)color.z;
  }
}

__global__ void raytraceRay(ray *rays, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, unsigned int numberOfGeoms, material *materials, unsigned int numberOfMaterials, light *lights, unsigned int numberOfLights)
{
  int rayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int pixelIndex = rays[rayIndex].pixelIndex;

  if(rayIndex <= cam.resolution.x * cam.resolution.y)
  {
	if(pixelIndex == -10000)
	{
		return;
	}

	if(rayDepth == traceDepth + 1)
	{
		colors[pixelIndex] = glm::vec3(0, 0, 0);
		return;
	}
	
	ray rt = rays[rayIndex];

	glm::vec3 intersectionPoint, normal;
	int geomIndex = findNearestPrimitiveInRay(geoms, numberOfGeoms, rt, intersectionPoint, normal);
	if(geomIndex != -1)
	{
		material m = materials[geoms[geomIndex].materialid];
		//Light Source
		if(m.emittance >= 1.0f)
		{
			colors[pixelIndex] *= (m.color * m.emittance);
			rays[rayIndex].pixelIndex = -10000;
		}

		//Non-light Object
		else
		{
			AbsorptionAndScatteringProperties absorptionAndScatteringProperties;
			thrust::default_random_engine rng(hash(time * rayDepth * rayIndex));
			thrust::uniform_real_distribution<float> u01(0,1);
			float randomNumber1 = (float)u01(rng);
			float randomNumber2 = (float)u01(rng);

			glm::vec3 color(1, 1, 1);
			rt = rays[rayIndex];
			glm::vec3 unabsorbed(0, 0, 0);
			calculateBSDF(rt, intersectionPoint, normal, glm::vec3(0, 0, 0), absorptionAndScatteringProperties, color, unabsorbed,
																m,	geoms, numberOfGeoms, geomIndex, randomNumber1, randomNumber2);
			rays[rayIndex] = rt;
			colors[pixelIndex] *= color;
		}
	}
	//Background
	else
	{
		colors[pixelIndex] = glm::vec3(0, 0, 0);
		rays[rayIndex].pixelIndex = -10000;
	}
  }
  //__syncthreads();
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,
	light* lights, int numberOfLights)
{
  // set up crucial magic

  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  //image for current frame
  glm::vec3 *currentFrame = NULL;
  cudaMalloc((void**)&currentFrame, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  
  //package geometry and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }

  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  //Packaging materials and sending to GPU
  material *cudaMaterials = NULL;
  cudaMalloc((void**)&cudaMaterials, numberOfMaterials * sizeof(material));
  cudaMemcpy(cudaMaterials, materials, numberOfMaterials * sizeof(material), cudaMemcpyHostToDevice);

  //Packaging lights and sending to GPU, used only in ray tracer
  light *cudaLights = NULL;
  //cudaMalloc((void**)&cudaLights, numberOfLights * sizeof(light));
  //cudaMemcpy(cudaLights, lights, numberOfLights * sizeof(light), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //Packaging rays
  int numberOfRays = (int)cam.resolution.x * (int)cam.resolution.y;
  ray *cudaRays = NULL;
  cudaMalloc((void**)&cudaRays, numberOfRays * sizeof(ray));

  //kernel launches
  initialRaysGenerator<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, cudaRays, currentFrame, iterations);

  //kernel launches
  unsigned int numberOfThreadsPerBlockTrace = 64;
  unsigned int numberOfBlocksTrace = (unsigned int)ceil((float)numberOfRays / numberOfThreadsPerBlockTrace);

  if(iterations == 1)
  {
	  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage);
  }

  int tempo = traceDepth / 2;

  for(int i = 1; i <= traceDepth + 1; ++i)
  {
	  raytraceRay<<<numberOfBlocksTrace, numberOfThreadsPerBlockTrace>>>(cudaRays, (float)iterations, cam, i, currentFrame, cudageoms, (unsigned int)numberOfGeoms, cudaMaterials,
		  (unsigned int)numberOfMaterials, cudaLights, (unsigned int)numberOfLights);
	  if(i % (tempo) == 0)
	  {
  		  thrust::device_ptr<ray> in_dev_ptr(cudaRays);
  		  thrust::device_ptr<ray> out_dev_ptr;
  		  out_dev_ptr = thrust::remove_if(in_dev_ptr, in_dev_ptr + numberOfRays, is_garbage_ray());
  		  numberOfRays = out_dev_ptr.get() - in_dev_ptr.get();
  		  numberOfBlocksTrace = (unsigned int)ceil((float)numberOfRays / numberOfThreadsPerBlockTrace);
	  }
  }

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, currentFrame, iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete [] geomList;

  //Freeing memory from materials, lights and rays
  cudaFree(cudaMaterials);
  //cudaFree(cudaLights);
  cudaFree(cudaRays);
  cudaFree(currentFrame);

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}


//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& rt, const glm::vec3 &intersectionPoint, const glm::vec3 &normal, glm::vec3 emittedColor, 
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, 
                                       glm::vec3& color, glm::vec3& unabsorbedColor, const material &m, const staticGeom const *geoms,
									   const int numberOfGeoms, const int geomIndex, const float randomNumber1, const float randomNumber2)
{
	//Deciding the nature of secondary ray
	int nextRay;	//	Diffuse -> 0, Reflect -> 1, Refract -> 2
	float reflectivity = m.hasReflective;
	float refractivity = m.hasRefractive;
	////Purely Diffuse
	//if(m.hasReflective <= 0.001f && m.hasRefractive <= 0.001f)
	//{
	//	nextRay = 0;
	//}
	////Has reflection / refraction
	//else
	//{
	//	float diffuseProb = (m.color.r + m.color.g + m.color.b) / 3.0f;
	//	//Reflective
	//	if(m.hasRefractive <= 0.001f)
	//	{
	//		float totalProb = diffuseProb + m.hasReflective;
	//		m.hasReflective /= totalProb;
	//		if(m.hasReflective > randomNumber1)
	//		{
	//			nextRay = 1;
	//		}
	//		else
	//		{
	//			nextRay = 0;
	//		}
	//	}
	//	//Refractive
	//	else if(m.hasReflective <= 0.001f)
	//	{
	//		float totalProb = diffuseProb + m.hasRefractive;
	//		m.hasRefractive /= totalProb;
	//		if(m.hasRefractive > randomNumber1)
	//		{
	//			nextRay = 2;
	//		}
	//		else
	//		{
	//			nextRay = 0;
	//		}
	//	}
	//	//Both Reflective and Refractive
	//	else
	//	{
	//		float totalProb = diffuseProb + m.hasReflective + m.hasRefractive;
	//		if(totalProb <= 0.001f)
	//		{
	//			nextRay = 0;
	//		}
	//		else
	//		{
	//			m.hasReflective /= totalProb;
	//			m.hasRefractive /= totalProb;
	//			diffuseProb /= totalProb;
	//			if(m.hasRefractive > randomNumber1)
	//			{
	//				nextRay = 2;
	//			}
	//			else if(m.hasReflective > randomNumber2)
	//			{
	//				nextRay = 1;
	//			}
	//			else
	//			{
	//				nextRay = 0;
	//			}
	//		}
	//	}
	//}

	float diffuseProb = (m.color.x + m.color.y + m.color.z) / 3.0f;
	float totalProb = diffuseProb + m.hasReflective + m.hasRefractive;
	if(totalProb <= 0.001f)
	{
		nextRay = 0;
	}
	else
	{
		reflectivity /= totalProb;
		refractivity /= totalProb;
		if(refractivity > randomNumber1)
		{
			nextRay = 2;
		}
		else if(reflectivity > randomNumber2)
		{
			nextRay = 1;
		}
		else
		{
			nextRay = 0;
		}
	}
	

	if(nextRay == 1)
	{
		float KR = 1.0f;   //For attenuating reflection
		glm::vec3 reflectedRay = calculateReflectionDirection(normal, rt.direction);
		rt.origin = intersectionPoint + reflectedRay * 0.01f;
		rt.direction = reflectedRay;

		//The diffuse component of object
		if(!(m.color.x <= 0.001f && m.color.y <= 0.001f && m.color.z <= 0.001f))
		{
			color *= m.color;
			color *= KR;
		}

		return 1;
	}
	else if(nextRay == 2)
	{
		float KRef = 1.0f;	//For attenuating refraction
		glm::vec3 reflectedRay = calculateReflectionDirection(normal, rt.direction);
		const float mediumRefractiveIndex = 1.0f;
		glm::vec3 refractedRayDirection = calculateTransmissionDirection(normal, rt.direction, mediumRefractiveIndex, m.indexOfRefraction);

		//This check is needed only if space is filled with a material of high-refractive index
		if(refractedRayDirection.x == -10000 && refractedRayDirection.y == -10000 && refractedRayDirection.z == -10000)
		{
			//Reflection, angle > critical angle
			rt.origin = intersectionPoint + reflectedRay * 0.01f;
			rt.direction = reflectedRay;
		}
		else
		{
			Fresnel fresnel = calculateFresnel(normal, rt.direction, mediumRefractiveIndex, m.indexOfRefraction,
																	refractedRayDirection);
			if(fresnel.transmissionCoefficient > randomNumber1)
			{
				ray refractedRay;
				refractedRay.direction = refractedRayDirection;
				refractedRay.origin = intersectionPoint + refractedRayDirection * 0.01f;
				glm::vec3 internalIntersectionPoint;
				glm::vec3 internalIntersectionNormal;
				int self = findNearestPrimitiveInRay(geoms, numberOfGeoms, refractedRay, internalIntersectionPoint, internalIntersectionNormal);
						
				//This will happen if eye is inside a primitive
				if(self != geomIndex)
				{
					rt.origin = intersectionPoint + refractedRayDirection * 0.01f;
					rt.direction = refractedRayDirection;
				}
				else
				{
					glm::vec3 refractedRayDirection2 = calculateTransmissionDirection(-internalIntersectionNormal, refractedRayDirection,
																				m.indexOfRefraction, mediumRefractiveIndex);
					if(refractedRayDirection2.x == -10000 && refractedRayDirection2.y == -10000 && refractedRayDirection2.z == -10000)
					{
						//TIR, ray gets trapped inside object
						//rays[rayIndex].pixelIndex = -10000;
						rt.direction = calculateReflectionDirection(-internalIntersectionNormal, refractedRayDirection);
						rt.origin = internalIntersectionPoint + rt.direction * 0.01f;
					}
					else
					{
						fresnel = calculateFresnel(-internalIntersectionNormal, refractedRayDirection, m.indexOfRefraction,	mediumRefractiveIndex, refractedRayDirection2);
						if(fresnel.transmissionCoefficient > randomNumber2)
						{
							rt.origin = internalIntersectionPoint + refractedRayDirection2 * 0.01f;
							rt.direction = refractedRayDirection2;
						}
						else
						{
							rt.direction = calculateReflectionDirection(-internalIntersectionNormal, refractedRayDirection);
							rt.origin = internalIntersectionPoint + rt.direction * 0.01f;
						}
					}
				}
			}
			else
			{
				rt.origin = intersectionPoint + reflectedRay * 0.01f;
				rt.direction = reflectedRay;
			}
		}

		//The diffuse component of object
		if(!(m.color.x <= 0.001f && m.color.y <= 0.001f && m.color.z <= 0.001f))
		{
			color *= m.color;
			color *= KRef;
		}

		return 2;
	}
	else
	{
		if(randomNumber1 > rt.survivalProbability)
		{
			rt.pixelIndex = -10000;
			return -1;
		}
		else
		{
			rt.direction = calculateRandomDirectionInHemisphere(normal, randomNumber1, randomNumber2);
			rt.origin = intersectionPoint + rt.direction * 0.01f;
			color *= m.color;
			if(color.x <= 0.001f && color.y <= 0.001f && color.z <= 0.001f)
			{
				rt.survivalProbability = 0.0f;
			}
			else
			{
				rt.survivalProbability *= 0.9f;
			}
		}

		return 0;
	}
}

__host__ __device__ int findNearestPrimitiveInRay(const staticGeom const *geoms, int numberOfGeoms, const ray &rt, glm::vec3 &intersectionPoint, glm::vec3 &normal) //Return -1 if no intersection, else index of geom
{
	float dist = 10000000.0f;
	int geomIndex = -1;
	for(unsigned int i = 0; i < numberOfGeoms; ++i)
	{
		glm::vec3 currentIntersectionPoint, currentNormal;
		float currentDist = 0.0f;
		if(geoms[i].type == SPHERE)
		{
			currentDist = sphereIntersectionTest(geoms[i], rt, currentIntersectionPoint, currentNormal);
			if(currentDist != -1.0f && currentDist < dist)
			{
				intersectionPoint = currentIntersectionPoint;
				normal = currentNormal;
				dist = currentDist;
				geomIndex = i;
			}
		}
		else if(geoms[i].type == CUBE)
		{
			currentDist = boxIntersectionTest(geoms[i], rt, currentIntersectionPoint, currentNormal);
			if(currentDist != -1.0f && currentDist < dist)
			{
				intersectionPoint = currentIntersectionPoint;
				normal = currentNormal;
				dist = currentDist;
				geomIndex = i;
			}
		}
		else if(geoms[i].type == MESH)
		{
			
		}
	}
	return geomIndex;
}