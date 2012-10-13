// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com
#include <thrust\copy.h>
#include <thrust\remove.h>
#include <thrust\count.h>
#include <thrust\device_vector.h>
#include <thrust\device_ptr.h>
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
#include "EasyBMP.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h> 



using namespace std;
#define MAX_DEPTH 100


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
   
  thrust::default_random_engine rng(hashF(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, int index){
	  ray r;


    thrust::default_random_engine rng( hashF(time * index) );
	thrust::uniform_real_distribution<float> u01(-0.5, 0.5); 
	float jitterX = u01(rng);
	float jitterY = u01(rng);
	    //standard camera raycast stuff
	  glm::vec3 e = eye;
	  glm::vec3 C = view;
	  glm::vec3 U = up;
	  float fovx = fov.x;
	  float fovy = fov.y;
  
	  float CD = glm::length(C);
  
	  glm::vec3 A = glm::cross(C, U);
	  glm::vec3 B = glm::cross(A, C);
	  glm::vec3 M = e+C;
	  glm::vec3 H = (A*float(CD*tan(fovx*(PI/180))))/float(glm::length(A));
	  glm::vec3 V = (B*float(CD*tan(-fovy*(PI/180))))/float(glm::length(B));
  
	  float sx = (x+jitterX)/(resolution.x-1);
	  float sy = (y+jitterY)/(resolution.y-1);
  
	  glm::vec3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
	  glm::vec3 PmE = P-e;
	  glm::vec3 R = e + (float(200)*(PmE))/float(glm::length(PmE));
  
	  glm::vec3 direction = glm::normalize(R);
	  r.origin = eye;
	  r.direction = direction;
	  	  

	  return r;



}

__global__ void initiateRays(glm::vec2 resolution, cameraData cam, ray *r,glm::vec3 * colors)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = -x + (y * resolution.x);
    if(x <= resolution.x && y <= resolution.y)
	{
		r[index] = raycastFromCameraKernel(cam.resolution, 0.0f, x, y, cam.position, cam.view, cam.up, cam.fov,index);
		r[index].colorIndex = index;
		r[index].hasStopped = false;
		colors[index] = glm::vec3(1);
		r[index].color = glm::vec3(1);
		r[index].reductionCoeficient = 1.0;
		r[index].isRayInside = false;
		r[index].IOR = 1.0;

	}
__syncthreads();
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

__host__ __device__ int findNearestIntersectionPoint(const staticGeom const *geoms, int numberOfGeoms, const ray &r, glm::vec3 &intersectionPoint, glm::vec3 &normal){

	            float closestIntersectionDistance = 100000.0;
				float hitPointDistance = -1;
				int currentIndex = -1;
				glm::vec3 tempNormal = glm::vec3(0);
				glm::vec3 tempIntersectionPoint = glm::vec3(0);
				
				for(int i = 0; i < numberOfGeoms; i++)
				{
						if( geoms[i].type == SPHERE )
						{
							hitPointDistance = sphereIntersectionTest(geoms[i], r, tempIntersectionPoint, tempNormal);

						}else if(geoms[i].type == CUBE)
						{
							hitPointDistance = boxIntersectionTest(geoms[i], r, tempIntersectionPoint, tempNormal);
					      
						}
						//find the closest intersetion point
						if(hitPointDistance > 0.0)
						{
							if(hitPointDistance < closestIntersectionDistance)
							{
								closestIntersectionDistance = hitPointDistance;
								normal = tempNormal;
								intersectionPoint = tempIntersectionPoint;
								currentIndex = i;
							}
						}
				}//end geoms for loop
				return currentIndex;

}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(ray * r, float time, int rayDepth, glm::vec3* colors, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, int numberOfThreads){

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int colorIndex = r[index].colorIndex;


		  if(index > numberOfThreads){
			  r[index].hasStopped = true;
			
			  return;
		  }
		  if( r[index].hasStopped){

			  return;
		  }
		  if(rayDepth > MAX_DEPTH && !r[index].hasStopped)
			{
				colors[colorIndex] = glm::vec3(0.0f, 0.0f, 0.0f);
				r[index].hasStopped = true;
				return;
			}

		  
		  glm::vec3 intersectionPoint = glm::vec3(0);
		  glm::vec3 normal = glm::vec3(0);
		  int currentIndex = findNearestIntersectionPoint(geoms,numberOfGeoms,r[index],intersectionPoint,normal);

		  //if no object is intersected then color is black
		  if(currentIndex == -1){
			  r[index].hasStopped = true;
			  colors[colorIndex] *= glm::vec3(0);
			  return;
		  }

		  if( materials[geoms[currentIndex].materialid].emittance >0){
			       r[index].hasStopped = true;			  
				   colors[colorIndex] *= materials[geoms[currentIndex].materialid].color * materials[geoms[currentIndex].materialid].emittance;
			   
		  }else  {
			    thrust::default_random_engine rng((hashF(time) * hashF(index) * hashF(rayDepth)));
			    thrust::uniform_real_distribution<float> u01(0,1);
			  
				float incidentIOR =r[index].IOR;
				float transmittedIOR = materials[geoms[currentIndex].materialid].indexOfRefraction;
								
				glm::vec3 reflectionDirection = calculateReflectionDirection(normal, r[index].direction);
				glm::vec3 transmissionDirection = calculateTransmissionDirection(normal,  r[index].direction, incidentIOR, transmittedIOR);
				Fresnel fresnel= calculateFresnel(normal, r[index].direction, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection);


				bool reflective = false;
				bool refractive = false;
				if((materials[geoms[currentIndex].materialid].hasReflective > 0.0)&&(materials[geoms[currentIndex].materialid].hasRefractive > 0.0)){
					if(abs(fresnel.transmissionCoefficient+fresnel.reflectionCoefficient )>1){
							reflective=true;
						}else{					
							float russianRoulette = (float)u01(rng);
							if(russianRoulette<0.5)
								reflective=true;
							else							
							    refractive=true;
						}
				 }else if(materials[geoms[currentIndex].materialid].hasReflective > 0.0){
							reflective=true;
						}else if(materials[geoms[currentIndex].materialid].hasRefractive > 0.0){
							refractive=true;
						}
		
				/********************reflective*******************************************************************************************/
				if ( reflective ) {//reflective
					//   r[index].reductionCoeficient = calculateFresnel(normal, r[index].direction, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient;
					 //  colors[colorIndex] *= materials[geoms[currentIndex].materialid].specularColor;// *  r[index].reductionCoeficient;
					   r[index].origin = intersectionPoint;
					   r[index].direction =  reflectionDirection;
					   return;
				
				} 
				/********************refractive*******************************************************************************************/
				if (refractive) {
					    float cosAngle = glm::dot( normal, r[index].direction);
						float n = incidentIOR / transmittedIOR;
						float secondTerm = 1.0f - n * n * (1.0f - cosAngle * cosAngle);
	
						if (secondTerm >= 0.0f)
						{
							r[index].direction =( (n * r[index].direction) - (n * cosAngle + sqrtf( secondTerm )) * normal);
						}else
							r[index].direction = glm::vec3(0);

						r[index].origin = intersectionPoint;
						r[index].isRayInside = !r[index].isRayInside;
						if(r[index].isRayInside){
						   r[index].IOR = materials[geoms[currentIndex].materialid].indexOfRefraction;
						}else{
						   r[index].IOR = 1.0f;

						}
						return;
					
						
				}
				/********************diffuse*******************************************************************************************/
				if(!reflective && !refractive) {//diffuse
				   	r[index].origin = intersectionPoint;
					r[index].direction = glm::normalize(calculateRandomDirectionInHemisphere(normal,(float)u01(rng), (float)u01(rng) ));
					colors[colorIndex] *= materials[geoms[currentIndex].materialid].color;
				}
			
		  }

		 
 
 
 //__syncthreads();
		
}
//adjust Color
__global__ void adjustColor(glm::vec2 resolution, glm::vec3* colors, glm::vec3* accImage, float iterations)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	colors[index] = (accImage[index] * (iterations - 1.0f) + colors[index]) / (float)iterations;

}



//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = MAX_DEPTH; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
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


  //package rays to create threads over rays instead of for each pixel
  int numberOfRays = (int)cam.resolution.x * (int)cam.resolution.y;
  ray *rays = new ray[numberOfRays];
  ray *cudaRays = NULL;
  cudaMalloc((void**)&cudaRays, numberOfRays * sizeof(ray));
  cudaMemcpy(cudaRays, rays, numberOfRays * sizeof(ray), cudaMemcpyHostToDevice);
    

  //trace the rays first
  initiateRays<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,cam, cudaRays,cudaimage);

 
  unsigned int threadsPerBlockBounce = (int)(tileSize*tileSize);
  unsigned int  fullBlocksBouncePerGrid = ((unsigned int)ceil(float(numberOfRays)/float(threadsPerBlockBounce)));
  
  int i = 0;
  while (i<traceDepth && numberOfRays > 0 ){
	  
	raytraceRay<<<fullBlocksBouncePerGrid, threadsPerBlockBounce>>>(cudaRays, (float)iterations, i, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials, numberOfRays);
	thrust::device_ptr<ray> in_ray_ptr(cudaRays);
	thrust::device_ptr<ray> out_ray_ptr = thrust::remove_if(in_ray_ptr, in_ray_ptr + numberOfRays, is_ray_stopped());
	numberOfRays = out_ray_ptr.get() - cudaRays;	
	fullBlocksBouncePerGrid = (unsigned int)ceil((float)numberOfRays / threadsPerBlockBounce);
	i++;
  }

  glm::vec3* accImage = NULL;
  cudaMalloc((void**)&accImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( accImage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  adjustColor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage, accImage, (float)iterations);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree(cudaRays);
  cudaFree( accImage );
  delete [] rays;
  delete [] geomList;
 
  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
