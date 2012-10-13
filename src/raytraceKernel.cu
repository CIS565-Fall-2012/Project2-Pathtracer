// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <thrust\remove.h>
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


 struct isNegative
	{
		__host__ __device__ 
		bool operator()(const ray & x) 
		{
			return !x.useful;
		}
	};

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
__host__ __device__ glm::vec3 generateRandomNumberFromThread2(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(-0.5,0.5);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Kernel that does the initial raycast from the camera and caches the result. "First bounce cache, second bounce thrash!"

//host and device mean compile one version on the CPU and one version on CPU
__global__ void raycastFromCameraKernel(glm::vec2 resolution, float time, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, ray* ray_bundle, glm::vec3* temp){
   
  float x = (blockIdx.x * blockDim.x) + threadIdx.x;
  float y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
  int index = x + (y * resolution.x);

  glm::vec3 rand=generateRandomNumberFromThread2(resolution,time,x,y);
   
  x=x+rand.x;
  y=y+rand.y;

 
  //standard camera raycast stuff
  glm::vec3 E1 = eye;
  glm::vec3 C = view;
  glm::vec3 U = up;
  float fovx = fov.x;
  float fovy = fov.y;
  
  float CD = glm::length(C);
  
  glm::vec3 A = glm::cross(C, U);
  glm::vec3 B = glm::cross(A, C);
  glm::vec3 M = E1+C;
  glm::vec3 H = (A*float(CD*tan(fovx*(PI/180))))/float(glm::length(A));
  glm::vec3 V = (B*float(CD*tan(-fovy*(PI/180))))/float(glm::length(B));
  
  float sx = (x)/(resolution.x-1);
  float sy = (y)/(resolution.y-1);
  
  glm::vec3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
  glm::vec3 PmE = P-E1;
  glm::vec3 R = E1 + (float(200)*(PmE))/float(glm::length(PmE));
  
  glm::vec3 direction = glm::normalize(R);
  //major performance cliff at this point, TODO: find out why!
  bool dof=true;
  //if (dof)
  
  glm::vec3 focuspoint= glm::vec3(0,0,0);
  glm::vec3 unitvec1= glm::normalize(focuspoint-eye);
  glm::vec3 unitvec2= glm::normalize(direction);
  float z_dist= focuspoint.z- eye.z;

  // using similar triangles
	
  float y_dist = ((z_dist)/(unitvec2.z))*unitvec2.y;
  float x_dist=  unitvec2.x *((z_dist)/(unitvec2.z));
  
  glm::vec3 rand1=generateRandomNumberFromThread2(resolution,time*11,x,y);
  glm::vec3 new_eye= glm::vec3(eye.x+(rand1.x)*2,eye.y+(rand1.y)*2,eye.z);

  glm::vec3 new_direction=glm::normalize(eye+glm::vec3(x_dist,y_dist,z_dist) - new_eye);  
  ray r;
  r.origin = new_eye;
  r.direction = new_direction;
  

  ray_bundle[index].direction=new_direction;
  ray_bundle[index].origin=new_eye;
  ray_bundle[index].index_ray=index;
  ray_bundle[index].useful = true;

  // Reset temp
   temp[index]= glm::vec3(1.0f,1.0f,1.0f);
  //printf("\nraypacket index %f",ray_bundle->index_ray );  // also to check if we did all the 640000 rays
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

// global means launched by CPU
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float time, ray* raybundle){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;  
	  glm::vec3 addingcolor;
	
	  color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

	 // color+=color;

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

//__global__ void dof(raypacket)

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials,
							glm::vec3* renderimage, ray* raybundle, int bounces, glm::vec3* temp)
{

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if (raybundle[index].useful)
  {
  glm::vec3 old_color= glm::vec3(0,0,0);
  old_color=renderimage[raybundle[index].index_ray] * (time-1);
  //glm::clamp(old_color,0.0f,1.0f);
  
  //colors[raybundle[index].index_ray]=renderimage[raybundle[index].index_ray];

	if((x<=resolution.x && y<=resolution.y))
	{
		float depth = 0;
		glm::vec3 normal;

		glm::vec3 random_direction;
		int Object_number;	
		float tmin=-1;
		Object_number=0;
		glm::vec3 POI;
		//printf("\n%f  index", raybundle[index].index_ray);
		for(int i=0; i<numberOfGeoms; i++)
		{
			glm::vec3 intersectionPoint;
			glm::vec3 intersectionNormal;
			if(geoms[i].type==SPHERE)
			{
				depth = sphereIntersectionTest(geoms[i], raybundle[index], intersectionPoint, intersectionNormal);
				if(depth>-EPSILON)
				{
					if (tmin<=EPSILON || depth<tmin+EPSILON)
					{
						tmin=depth;
						POI=intersectionPoint;
						normal=intersectionNormal;
						Object_number=i;
					}
				}
			}
			else if(geoms[i].type==CUBE)
			{
				depth = boxIntersectionTest(geoms[i], raybundle[index], intersectionPoint, intersectionNormal);
				if (depth>-EPSILON)
				{
					if (tmin<=EPSILON || depth<tmin+EPSILON)
					{
						tmin=depth;
						POI=intersectionPoint;
						normal=intersectionNormal;
						Object_number=i;
					}
				}
			}
		}
				
		if(tmin<EPSILON)
		{
			raybundle[index].useful=false;
			temp[raybundle[index].index_ray]= glm::vec3(0,0,0);
			colors[raybundle[index].index_ray]=(old_color)/time; //old_color+
		}
		else
		{
			if (materials[geoms[Object_number].materialid].emittance>0 )
			{
			//	//glm::vec3 old_color=colors[index]*(time-1);
				colors[raybundle[index].index_ray]=(old_color+( temp[raybundle[index].index_ray]*materials[geoms[Object_number].materialid].color*materials[geoms[Object_number].materialid].emittance))/time; //old_color+
				//glm::clamp(colors[raybundle[index].index_ray],0.0f,1.0f);
				raybundle[index].useful=false;
				return;
			}
			 else if (materials[geoms[Object_number].materialid].hasReflective>0)
			{
				normal=glm::normalize(normal);
				raybundle[index].direction=glm::normalize(raybundle[index].direction);
				glm::vec3 reflectedRay;
				reflectedRay= raybundle[index].direction- (normal +normal)*(glm::dot(raybundle[index].direction,normal));
				reflectedRay = glm::normalize(reflectedRay);
				raybundle[index].origin=POI;
				raybundle[index].direction=reflectedRay;
			}
			 else if(materials[geoms[Object_number].materialid].hasRefractive>0)
			 {
				 float n1=1;
				 float n2=(materials[geoms[Object_number].materialid].indexOfRefraction);
				 glm::vec3 incident=raybundle[index].direction;

				 float magnitugeAxB=glm::length(glm::cross(normal,incident));
				 float magA=glm::length(normal);
				 float magB=glm::length(raybundle[index].direction);
				 float sin0i=glm::sin((magnitugeAxB)/(magA*magB));
				// printf("%f",sin0i);

				 float sin0r= ((n1)/(n2)) *sin0i;

				 float cos0i= glm::sqrt(1-sin0i*sin0i);
				 float cos0r= glm::sqrt(1-sin0r*sin0r);

				 glm::vec3 refracted= ((n1)/(n2))*incident + (((n1)/(n2))*cos0i-cos0r)*normal;

				 ray refractedray;
				 refractedray.origin=POI;
				 refractedray.direction= refracted;

				 for(int i=0; i<numberOfGeoms; i++)
				{
					glm::vec3 intersectionPoint;
					glm::vec3 intersectionNormal;
					if(geoms[i].type==SPHERE)
					{
						depth = sphereIntersectionTest(geoms[i], refractedray, intersectionPoint, intersectionNormal);
						if(depth>-EPSILON)
						{
							if (tmin<=EPSILON || depth<tmin+EPSILON)
							{
								tmin=depth;
								POI=intersectionPoint;
								normal=intersectionNormal;
								Object_number=i;
							}
						}
					}
					else if(geoms[i].type==CUBE)
					{
						depth = boxIntersectionTest(geoms[i], refractedray, intersectionPoint, intersectionNormal);
						if (depth>-EPSILON)
						{
							if (tmin<=EPSILON || depth<tmin+EPSILON)
							{
								tmin=depth;
								POI=intersectionPoint;
								normal=intersectionNormal;
								Object_number=i;
							}
						}
					}
				}

				 // when the light is going out of object
				  n2=1;
				  n1=(materials[geoms[Object_number].materialid].indexOfRefraction);
				  incident=refracted;

				  magnitugeAxB=glm::length(glm::cross(normal,incident));
				  magA=glm::length(normal);
				  magB=glm::length(refracted);
				  sin0i=glm::sin((magnitugeAxB)/(magA*magB));
				// printf("%f",sin0i);

				 sin0r= ((n1)/(n2)) *sin0i;

				 cos0i= glm::sqrt(1-sin0i*sin0i);
				 cos0r= glm::sqrt(1-sin0r*sin0r);

				 refracted= ((n1)/(n2))*incident - (((n1)/(n2))*cos0i-cos0r)*normal;

				 raybundle[index].direction=refracted;
				 raybundle[index].direction =(raybundle[index].direction);
				 raybundle[index].origin=POI;
				 
			 }
			else
			{
		 		glm::vec3 random1_num=generateRandomNumberFromThread(resolution,time*(bounces),x,y);
				random_direction=calculateRandomDirectionInHemisphere(normal,random1_num.x,random1_num.y);
				//random_direction=random_direction*glm::vec3(10,10,10);
				raybundle[index].direction=(random_direction);
				raybundle[index].origin=POI;
			 

				if (bounces==1)
				{
					 temp[raybundle[index].index_ray]=materials[geoms[Object_number].materialid].color;
					
				}
				else //if (bounces==2 && bounces <9)
				{
					 temp[raybundle[index].index_ray]= temp[raybundle[index].index_ray]*materials[geoms[Object_number].materialid].color; 
				}
			}
		}
	}
  }
}



//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces
  int bounces=0;
  int activerays=(int)renderCam->resolution.x*(int)renderCam->resolution.y;

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  

  //send image to GPU
 
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  glm::vec3* renderimage=NULL;
  cudaMalloc((void**)&renderimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy(renderimage,renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  glm::vec3* temp_color=NULL;
  cudaMalloc((void**)&temp_color,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  
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
	newStaticGeom.tranposeTranform=geoms[i].tranposeTranforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  

  ray* raypacket=NULL;
  cudaMalloc((void**)&raypacket,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));
  //cudaMemcpy(ray_packet, raypacket,640000*sizeof(ray),cudaMemcpyDevicetoDevice;


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
  //cam.image=renderCam->image;
  
  //kernel call for camera rays
  
	raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, (float)iterations, cam.position, cam.view, cam.up, cam.fov, raypacket, temp_color);

	//kernel launches
	while(bounces<10)
	{
		bounces++;
		dim3 fullBlockPerGridSC((int)ceil(float(renderCam->resolution.x)/float(tileSize)), ((int)activerays)/(float(tileSize)*(int)ceil(float(renderCam->resolution.x))));
	   
		raytraceRay<<<fullBlockPerGridSC, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, 
	   																			numberOfMaterials,renderimage, raypacket,bounces,temp_color);
		thrust::device_ptr<ray> devicePointer(raypacket);
		thrust::device_ptr<ray> newEnd=thrust::remove_if(devicePointer,devicePointer+activerays, isNegative());
		activerays= newEnd.get() - devicePointer.get();
	   
	}
  

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage,(float)iterations,raypacket);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //cudaMemcpy( renderCam->image, renderimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree(raypacket);
  cudaFree(cudaimage);
  cudaFree(cudageoms);
  cudaFree(cudamaterials);
  cudaFree(renderimage);
  cudaFree(temp_color);
  delete [] geomList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
//thrust::device_ptr<int> devicePointer(activePixels);
	//	thrust::device_ptr<int> newEnd = thrust::remove_if(devicePointer, devicePointer + numActivePixels, isNegative());
	//	numActivePixels = newEnd.get() - activePixels;