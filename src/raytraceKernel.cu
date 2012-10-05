// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
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

  //Point in space towards which ray has to be shot
  glm::vec3 p = m + (2.0f * x / (resolution.x - 1.0f) - 1.0f) * a + (2.0f * y / (resolution.y - 1.0f) - 1.0f) * b;
  r.direction = p - eye;
  r.direction = glm::normalize(r.direction);

  return r;
}

__global__ void initialRaysGenerator(cameraData cam, ray *rays)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * cam.resolution.x);
    if(x <= cam.resolution.x && y <= cam.resolution.y)
	{
		rays[index] = raycastFromCameraKernel(cam.resolution, 0.0f, x, y, cam.position, cam.view, cam.up, cam.fov);
    }
	__syncthreads();
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(const glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, const glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
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

__global__ void raytraceRay(ray *rays, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, unsigned int numberOfGeoms, material *materials, unsigned int numberOfMaterials, light *lights, unsigned int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int rayIndex = x + (y * cam.resolution.x);
  int index = (cam.resolution.x - x) + ((cam.resolution.y - y) * cam.resolution.x);

  if(((float)x <= cam.resolution.x && (float)y <= cam.resolution.y))
  {
	//ray rt = raycastFromCameraKernel(cam.resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
	ray rt = rays[rayIndex];
	if(rayDepth >= 10 || (rays[rayIndex].origin == glm::vec3(-10000, -10000, -10000)))
	{
		return;
	}

	glm::vec3 intersectionPoint, normal;
	int geomIndex = findNearestPrimitiveInRay(geoms, numberOfGeoms, rt, intersectionPoint, normal);
	if(geomIndex != -1)
	{
		//Flat Shading
		//colors[index] = materials[geoms[geomIndex].materialid].color;
		
		//Setting initial value
		if(rayDepth == 0)
		{
			colors[index] = glm::vec3(0, 0, 0);
		}
		for(unsigned int i = 0; i < numberOfLights; ++i)
		{
			ray lightRay;
			lightRay.origin = intersectionPoint;
			lightRay.direction = lights[i].position - intersectionPoint;
			if(glm::length(lightRay.direction) > 0.001f)
			{
				lightRay.direction = glm::normalize(lightRay.direction);
			}
			else
			{
				continue;
			}
			lightRay.origin += lightRay.direction * 0.01f;
			int obstructionIndex = -1;
			glm::vec3 obstructionIntersectionPoint, obstructionNormal;
			obstructionIndex = findNearestPrimitiveInRay(geoms, numberOfGeoms, lightRay, obstructionIntersectionPoint, obstructionNormal);
			if(obstructionIndex == -1 || (glm::distance(intersectionPoint, obstructionIntersectionPoint) > glm::distance(intersectionPoint, lights[i].position)))
			{
				//Lambert Shading
				float KD = 0.8f;
				colors[index] += KD * lights[i].color * materials[geoms[geomIndex].materialid].color * glm::dot(lightRay.direction, normal);
				//Phong Shading
				float KS = 0.10f;
				glm::vec3 reflectedRay = calculateReflectionDirection(normal, rt.direction);
				glm::vec3 V = glm::normalize((cam.position - intersectionPoint));
				colors[index] += (KS * materials[geoms[geomIndex].materialid].specularColor * lights[i].color * pow((float)glm::dot(reflectedRay, V),
					materials[geoms[geomIndex].materialid].specularExponent));
				//Reflection
				if(materials[geoms[geomIndex].materialid].hasReflective == 1.0f)
				{
					rays[rayIndex].origin = intersectionPoint + reflectedRay * 0.01f;
					rays[rayIndex].direction = reflectedRay;
				}
				//Refraction
				else if(materials[geoms[geomIndex].materialid].hasRefractive == 1.0f)
				{

				}
				else
				{
					rays[rayIndex].origin = glm::vec3(-10000, -10000, -10000);
				}
			}
			//Coloring due to reflection
			else if(materials[geoms[geomIndex].materialid].hasReflective == 1.0f)
			{
				glm::vec3 reflectedRay = calculateReflectionDirection(normal, rt.direction);
				rays[rayIndex].origin = intersectionPoint + reflectedRay * 0.01f;
				rays[rayIndex].direction = reflectedRay;
			}
			//Coloring due to refraction
			else if(materials[geoms[geomIndex].materialid].hasRefractive == 1.0f)
			{

			}
			//Ambient Lighting
			float KA = 0.1f;
			glm::vec3 ambientLight(0.2f, 0.2f, 0.2f);
			colors[index] += KA * materials[geoms[geomIndex].materialid].color * ambientLight;
		}
	}
	//Background
	else
	{
		glm::vec3 backRGB(0, 0, 0);
		colors[index] = backRGB;
	}
  }
  //__syncthreads();
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,
	light* lights, int numberOfLights){
  
  unsigned int traceDepth = 10; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
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

  //Packaging lights and sending to GPU
  light *cudaLights = NULL;
  cudaMalloc((void**)&cudaLights, numberOfLights * sizeof(light));
  cudaMemcpy(cudaLights, lights, numberOfLights * sizeof(light), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //Packaging rays
  int numberOfRays = (int)cam.resolution.x * (int)cam.resolution.y;
  ray *rays = new ray[numberOfRays];
  ray *cudaRays = NULL;
  cudaMalloc((void**)&cudaRays, numberOfRays * sizeof(ray));
  cudaMemcpy(cudaRays, rays, numberOfRays * sizeof(ray), cudaMemcpyHostToDevice);

  //kernel launches
  initialRaysGenerator<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, cudaRays);

  //kernel launches
  for(int i = 0; i < traceDepth; ++i)
  {
	  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaRays, (float)iterations, cam, i, cudaimage, cudageoms, (unsigned int)numberOfGeoms, cudaMaterials,
		  (unsigned int)numberOfMaterials, cudaLights, (unsigned int)numberOfLights);
  }

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  //delete geomList;
  delete [] geomList;

  //Freeing memory from materials, lights and rays
  cudaFree(cudaMaterials);
  cudaFree(cudaLights);
  cudaFree(cudaRays);
  delete [] rays;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
