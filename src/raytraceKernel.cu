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
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include "glm/glm.hpp"
#include <thrust/remove.h>
#include <thrust/device_ptr.h> 
#include "glm/gtx/vector_access.hpp"

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
   
  thrust::default_random_engine rng(hash(index)*hash(time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Kernel that does the initial raycast from the camera and caches the result. "First bounce cache, second bounce thrash!"
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
   
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index)*hash(time));
  thrust::uniform_real_distribution<float> u01(0,1);
  
  //standard camera raycast stuff
  glm::vec3 Eye = eye;
  glm::vec3 C = view;
  glm::vec3 U = up;
  float fovx = fov.x;
  float fovy = fov.y;
  
  float CD = glm::length(C);
  
  glm::vec3 A = glm::cross(C, U);
  glm::vec3 B = glm::cross(A, C);
  glm::vec3 M = Eye+C;
  glm::vec3 H = (A*float(CD*tan(fovx*(PI/180))))/float(glm::length(A));
  glm::vec3 V = (B*float(CD*tan(-fovy*(PI/180))))/float(glm::length(B));

  // generate a small shake to acheive Anti-Aliasing
  float dx = (float)u01(rng) - 0.5f;
  float dy = (float)u01(rng) - 0.5f;
  
  float sx = (x+dx)/(resolution.x-1);
  float sy = (y+dy)/(resolution.y-1);
   
  glm::vec3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
   P = Eye + glm::normalize((P - Eye))*5.f;

#ifdef RAYTRACEKERNEL_DEPTH_OF_FIELD 
  float lenseRadius = 0.05f;
  float randomAngle = (float)u01(rng) * TWO_PI;

  float randomDistance = lenseRadius * (float)u01(rng);
  
  float cam_dx = randomDistance * cos(randomAngle);
  float cam_dy = randomDistance * sin(randomAngle);

  Eye += cam_dx*glm::normalize(H) + cam_dy*glm::normalize(V);	
#endif
  
  glm::vec3 PmE = P-Eye;
  //glm::vec3 R = Eye + (float(200)*(PmE))/float(glm::length(PmE));
 // glm::vec3 R = Eye + PmE/glm::length(PmE);
  
  glm::vec3 direction = glm::normalize(PmE);
  //major performance cliff at this point, TODO: find out why!
  ray r;
  r.origin = eye;
  r.direction = direction;
  return r;
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, int numOfIters){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){
	  float oneOverIters = 1.f/(float)numOfIters;

      glm::vec3 color;      
      color.x = image[index].x*oneOverIters*255.0;
      color.y = image[index].y*oneOverIters*255.0;
      color.z = image[index].z*oneOverIters*255.0;

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
__global__ void raytraceRay(float time,
                            int rayDepth,
                            glm::vec3* colors, 
                            const staticGeom* geoms, int numberOfGeoms,
                            const material* materials, int numberOfMaterials,
                            PixelRay* pixelRays, int numOfRays,
                            glm::vec3* acc_refl_diff_colors)
{
	const int index = blockDim.x*blockIdx.x + threadIdx.x;
	PixelRay* const pixRay = &pixelRays[index];

	if (index < numOfRays) {
		const int pixID = pixRay->pixelID;
		glm::vec3 intersectionPoint, normal;
		float intersectionDistance;
		int intersectionGeomInd = findClosestIntersection(geoms, numberOfGeoms, pixRay->r,
			&intersectionPoint, &normal, &intersectionDistance); 

		if (intersectionGeomInd == -1) { // no hit!
			// terminate this ray
			pixRay->pixelID = -1;
			return;
		}

		const material objectMaterial = materials[geoms[intersectionGeomInd].materialid];

		if (objectMaterial.emittance > 0.f) { // light source
			colors[pixID] += acc_refl_diff_colors[pixID] * objectMaterial.emittance * objectMaterial.color;
			pixRay->pixelID = -1;
			return;
		}    
		// diffuse materials
		thrust::default_random_engine rng(hash(pixID)*hash(time)*hash(rayDepth));
		thrust::uniform_real_distribution<float> u01(0,1);   
		int diffOrSpec = decideDiffOrSpec(objectMaterial.hasReflective, (float)u01(rng));

		if (diffOrSpec == 1) { // 1. sepcular reflection
			pixRay->r.direction = glm::normalize(calculateReflectionDirection(normal, pixRay->r.direction));
			pixRay->r.origin = intersectionPoint + RAY_BIAS_AMOUNT*pixRay->r.direction;
			return;
		}

		// 2. diffuse reflection
		//colors[pixID] += acc_refl_diff_colors[pixID] * objectMaterial.emittance;
		acc_refl_diff_colors[pixID] *= objectMaterial.color;

		pixRay->r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal,
			(float)u01(rng),
			(float)u01(rng)));
		pixRay->r.origin = intersectionPoint + RAY_BIAS_AMOUNT*pixRay->r.direction;
		return;
	}
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
   const int numOfPixels = (int)renderCam->resolution.x * (int)renderCam->resolution.y;

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, numOfPixels*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, numOfPixels*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  //package geometry and materials and sent to GPU
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
  
  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // prepare some data structures for each pixel
  // 1. a random PixelRay, 2. an accumulated reflective and diffuse colors
  cameraData* cuda_cam = NULL;
  cudaMalloc((void**)&cuda_cam, sizeof(cameraData));
  cudaMemcpy(cuda_cam, &cam, sizeof(cameraData), cudaMemcpyHostToDevice);

  PixelRay* cuda_pixelRays = NULL;
  cudaMalloc((void**)&cuda_pixelRays, numOfPixels*sizeof(PixelRay));

  glm::vec3* cuda_acc_refl_diff_colors = NULL;  
  cudaMalloc((void**)&cuda_acc_refl_diff_colors, numOfPixels*sizeof(glm::vec3));

  preparePathTracing<<<fullBlocksPerGrid, threadsPerBlock>>>(cuda_cam, (float)iterations,
                                                            cuda_pixelRays, cuda_acc_refl_diff_colors);
  
  int numOfAliveRays = numOfPixels; 
  thrust::device_ptr<PixelRay> ray_end_ptr;
  const int rayThreadsPerBlock = 256; //TODO: play with different numbers
  for (int i = 0; numOfAliveRays > 0 && i < RAYTRACEKERNEL_RAY_BOUNCE_MAX; i++) {
    const int rayBlocksPerGrid = (int)ceil((float)(numOfAliveRays+1)/rayThreadsPerBlock);
    
    //kernel launches
    raytraceRay<<<rayBlocksPerGrid, rayThreadsPerBlock>>>((float)iterations,
														  i,
                                                          cudaimage,
                                                          cudageoms, numberOfGeoms,
                                                          cudamaterials, numberOfMaterials,
                                                          cuda_pixelRays, numOfAliveRays,
                                                          cuda_acc_refl_diff_colors);
	checkCUDAError("2Kernel failed!");

    // perform stream comopaction on PixelRays
	thrust::device_ptr<PixelRay> cuda_pixelRays_devPtr(cuda_pixelRays);
    ray_end_ptr =
        thrust::remove_if(cuda_pixelRays_devPtr, cuda_pixelRays_devPtr + numOfAliveRays, isTerminated());
    numOfAliveRays = (int)(ray_end_ptr - cuda_pixelRays_devPtr);
  }

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree( cuda_pixelRays );
  cudaFree( cuda_acc_refl_diff_colors );
  cudaFree( cuda_cam );
  cudaFree( cudaimage );
  delete [] geomList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}


// kernel to prepare some data structures needed by PathTracing
__global__ void preparePathTracing(const cameraData* cam, float time,
                                 PixelRay* pixelRays, glm::vec3* acc_refl_diff_colors)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * cam->resolution.x);

  if (index < (int)cam->resolution.x*cam->resolution.y) {
    // 1. pixel rays
    pixelRays[index].r =
        raycastFromCameraKernel(cam->resolution, time, x, y, cam->position, cam->view, cam->up, cam->fov);
    pixelRays[index].pixelID = index;

    // 2. initialize the accumulator for reflective and diffuse colors
	glm::set(acc_refl_diff_colors[index], 1.f, 1.f, 1.f);
  }
}


//// kernel to accumulate colors acheived in this iteration to the total accumulator
//__global__ void accumulateColors(const glm::vec2 resolution, glm::vec3* totalAccumulator, const glm::vec3* accuThisIter)
//{
//	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//	int index = x + (y * resolution.x);
//
//	if (index < (int)resolution.x*resolution.y) {
//		totalAccumulator[index] += accuThisIter[index];
//	}
//}