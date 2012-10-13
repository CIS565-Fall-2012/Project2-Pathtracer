// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com


#include <thrust/device_ptr.h>
#include <thrust\remove.h>

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

//Kernel that does the initial raycast from the camera and caches the result. "First bounce cache, second bounce thrash!"
__global__ void raycastFromCameraKernel(glm::vec2 resolution, float time, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, ray* r){
   

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = -x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);
  
  //standard camera raycast stuff
  glm::vec3 E = eye;
  glm::vec3 C = view;
  glm::vec3 U = up;
  float fovx = fov.x;
  float fovy = fov.y;
  
  float CD = glm::length(C);
  
  glm::vec3 A = glm::cross(C, U);
  glm::vec3 B = glm::cross(A, C);
  glm::vec3 M = E+C;
  glm::vec3 H = (A*float(CD*tan(fovx*(3.14/180))))/float(glm::length(A));
  glm::vec3 V = (B*float(CD*tan(-fovy*(3.14/180))))/float(glm::length(B));
  
  float sx = (x)/(resolution.x-1);
  float sy = (y)/(resolution.y-1);
  
  glm::vec3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
  glm::vec3 PmE = P-E;
  glm::vec3 R = E + (float(200)*(PmE))/float(glm::length(PmE));
  
  glm::vec3 direction = glm::normalize(R);
  //major performance cliff at this point, TODO: find out why!

  r[index].origin = eye;
  r[index].direction = direction;
  r[index].color = glm::vec3(1,1,1);
  r[index].index = index;
  r[index].hasStopped = false;
  r[index].hitLight = false;
  r[index].isInside = false;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, ray* r)
{

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = /*resolution.x * resolution.y - */(-x + (y * resolution.x));

	material curMaterial;
	
	if((x<=resolution.x && y<=resolution.y) && r[index].hasStopped == false)
	{
		glm::vec3 intersectionPoint;
		glm::vec3 intersectionNormal;
		float MAX_DEPTH = 100000000000000000;
		float depth = MAX_DEPTH;

		//intersection tests
		for(int i=0; i<numberOfGeoms; i++)
		{        
			if(geoms[i].type==SPHERE)
			{
				depth = sphereIntersectionTest(geoms[i], r[index], intersectionPoint, intersectionNormal);
			}
			else if(geoms[i].type==CUBE)
			{
				depth = boxIntersectionTest(geoms[i], r[index], intersectionPoint, intersectionNormal);
			}

			if(depth<MAX_DEPTH && depth>-0.001)
			{
				MAX_DEPTH = depth;
				curMaterial = materials[geoms[i].materialid];
			}
		}

		//if no object is hit
		if(MAX_DEPTH == 100000000000000000)
		{
			r[index].color = glm::vec3(0,0,0);
			
			r[index].hasStopped = true;
		}

		else
		{
			//if object is light source
			if(curMaterial.emittance > 0)
			{
				r[index].color  *= curMaterial.emittance * curMaterial.color;
				
				r[index].hasStopped = true;
				r[index].hitLight = true;					
			}

			else
			{
				r[index].origin = intersectionPoint;
				
				//reflective
				if(curMaterial.hasReflective > 0)
				{
					float russianRoulette = generateRandomNumberFromThread(resolution, time * rayDepth, x, y).x;
					if((float)russianRoulette > 0.5)
					{
						r[index].color *= curMaterial.color;
						r[index].direction = calculateReflectionDirection(intersectionNormal, r[index].direction);
					}
					else
					{
						r[index].color *= curMaterial.color;
						r[index].direction = calculateRandomDirectionInHemisphere(intersectionNormal, generateRandomNumberFromThread(resolution, time * rayDepth, x, y).x, generateRandomNumberFromThread(resolution, time * rayDepth, x, y).y);				
						glm::normalize(r[index].direction);
					}
				}
				
				//refractive
				else if(curMaterial.hasRefractive > 0)
				{
					if(r[index].isInside == false)
					{
						r[index].direction = calculateTransmissionDirection(glm::normalize(intersectionNormal), glm::normalize(r[index].direction), 1.0f, curMaterial.indexOfRefraction);
						glm::normalize(r[index].direction);
						r[index].isInside = true;
					}
					else
					{
						r[index].direction = calculateTransmissionDirection(glm::normalize(intersectionNormal), glm::normalize(r[index].direction), curMaterial.indexOfRefraction, 1.0f);
						glm::normalize(r[index].direction);
						r[index].isInside = false;
					}
				}

				//diffuse
				else
				{
					r[index].color *= curMaterial.color;
					r[index].direction = calculateRandomDirectionInHemisphere(intersectionNormal, generateRandomNumberFromThread(resolution, time * rayDepth, x, y).x, generateRandomNumberFromThread(resolution, time * rayDepth, x, y).y);				
					glm::normalize(r[index].direction);
				}
			}
		} //end else (object is hit)

		clamp(r[index].color.x, 0.0f, 1.0f);
		clamp(r[index].color.y, 0.0f, 1.0f);
		clamp(r[index].color.z, 0.0f, 1.0f);

		if(r[index].hitLight == true)
			colors[r[index].index] = ((time - 1) * colors[r[index].index] + r[index].color) / time;
	
	} // end if (x and y)
}


struct has_stopped// : public thrust::unary_function<ray,bool>
{
    __host__ __device__
    bool operator()(ray r)
    {
		return r.hasStopped;
    }
};


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
	int traceDepth = 10; //determines how many bounces the raytracer traces
	int size = (int)renderCam->resolution.x*(int)renderCam->resolution.y;

	// set up crucial magic
	glm::vec3* cudaimage = NULL;
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	staticGeom* cudageoms = NULL;
	material* cudamaterials = NULL;
	//ray* rayList = new ray[(int)renderCam->resolution.x * (int)renderCam->resolution.y];
	ray* cudarays = NULL;
	
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	//send image to GPU
	//glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
	//package geometry and materials and sent to GPU
	//staticGeom* geomList = new staticGeom[numberOfGeoms];
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
  
	//staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
	//material* cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	//send rays to GPU
	cudaMalloc((void**)&cudarays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));
	//cudaMemcpy( cudarays, rayList, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray), cudaMemcpyHostToDevice);

	//package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;

	raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam.position, cam.view, cam.up, cam.fov, cudarays);
	
	while(traceDepth > 0)
	{
		dim3 newthreadsPerBlock(tileSize, tileSize);
		dim3 newfullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), ((int)ceil(float(size)/(int)ceil(float(renderCam->resolution.x))))/float(tileSize));
		//kernel launches
		raytraceRay<<<newfullBlocksPerGrid, newthreadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, 
																					numberOfMaterials, cudarays);		
		//stream compaction

		thrust::device_ptr<ray> start(cudarays);//, new_end;
		thrust::device_ptr<ray> new_end = thrust::remove_if(start, start + size, has_stopped());
		size = new_end - start;

		traceDepth--;	
	} // end of while (traceDepth)
	
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	//free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudamaterials );
	cudaFree( cudarays );
	delete [] geomList;
	
	// make certain the kernel has completed 
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
