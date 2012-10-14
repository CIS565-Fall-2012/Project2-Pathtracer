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
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/device_allocator.h>
#include <thrust/device_new_allocator.h>

//Define Max Depth Here
#define MAX_DEPTH 8

//This variable is used to decide the depths at which stream compact will be run. For eg,if it is 3, then stream compaction will run every 3rd depth. Setting it to 0 will turn stream compaction off.
#define StreamCompactDepth 1

//Comment this line to turn depth of field off
//Uncomment the line to turn depth of field on
//#define USE_DEPTH_OF_FIELD	

//Comment this line to turn off Anti-aliasing
//Uncomment the line to turn on Anti-aliasing
#define USE_ANTI_ALIASING

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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;
  //r.origin = glm::vec3(0,0,0);
  //r.direction(0.0f,0.0f,-1.0f);
  r.origin = eye;
  glm::vec3 Avec = glm::cross(view, up);
  glm::vec3 Bvec = glm::cross(Avec, view);
  glm::vec3 Mvec = eye + view;
  glm::vec3 Hvec = (Avec * (float)(view.length() * tan(fov.x))) / (float) Avec.length();
  glm::vec3 Vvec = (Bvec * (float)(view.length() * tan(fov.y))) / (float) Bvec.length();
  float sx = (float)x / (float) resolution.x;
  float sy = (float)y / (float) resolution.y;
  glm::vec3 P = Mvec + (Hvec * (float)(2.0f*sx -1)) + (Vvec * (float)(2.0 * sy - 1));
  r.direction = glm::normalize(P - eye);

  return r;
}

//Get initial rays using kernels
__global__ void GetRayCastFromCameraKernel(cameraData cam, glm::vec3* InitCamVecs, float time, ray* InitialRays, float FocalDistance) //glm::vec3 FocusPoint)
//__global__ void GetRayCastFromCameraKernel(cameraData cam, float time, ray* InitialRays)
{

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);
	
#ifdef USE_ANTI_ALIASING
	//Generate anti-aliasing random values
	thrust::default_random_engine rng(hash(index*time + 100));
	thrust::uniform_real_distribution<float> X(-0.5, 0.5);
	float u = X(rng);
	float v = X(rng);
	
	float sx = ((float)x + u) / (float) cam.resolution.x;
	float sy = ((float)y + v) / (float) cam.resolution.y;

#else

	float sx = ((float)x) / (float) cam.resolution.x;
	float sy = ((float)y) / (float) cam.resolution.y;

#endif

	glm::vec3 P = InitCamVecs[0] + (InitCamVecs[1] * (float)(2.0f*sx -1)) + (InitCamVecs[2] * (float)(2.0 * sy - 1));

	ray r;
	r.origin = cam.position;
	
	r.direction = glm::normalize(P - r.origin);

#ifdef USE_DEPTH_OF_FIELD
	//Find Point on Plane
	//float D = (FocusPoint.z - r.origin.z) / r.direction.z;
	glm::vec3 PoP = r.origin + r.direction * FocalDistance;

	//jitter camera in X and Y
	/////SET RANGE OF CAMERA JITTER HERE////
	thrust::uniform_real_distribution<float> X1(-0.5, 0.5);
	float Xj = X1(rng);
	float Yj = X1(rng);

	r.origin += glm::vec3(Xj, Yj, 0.0f);
	r.direction = glm::normalize(PoP - r.origin);
#endif

	r.keep = 1;
	//r.newIndex = 1;
	r.pixelIndex = index;
	r.isInside = false;
	r.currentObjIndex = -1;
	r.currentMatIndex = -1;
	InitialRays[index] = r;
}

//Copy the Keep values of Rays into a device vector
__global__ void CopyKeepIntoVector(glm::vec2 resolution, ray* InitialRays, int* KeepArrayPointer)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	
	KeepArrayPointer[index] = InitialRays[index].keep;
}

//Kernel that whites out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
	  image[index] = glm::vec3(1,1,1);
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

//Accumalate Color
__global__ void AccumalateColor(glm::vec2 resolution, glm::vec3* colors, glm::vec3* cameraImage, float iterations)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	float toneMap = 1.0f / 1.2f;
	colors[index] = glm::vec3(pow(colors[index].x, toneMap), pow(colors[index].y, toneMap), pow(colors[index].z, toneMap)); 
	colors[index] = (cameraImage[index] * (iterations - 1.0f) + colors[index]) / (float)iterations;
	
}

//This function checks intersections. Intersection point is the real world point of intersection. Index is the index of the closest object. Returns true if there is intersections, false if there is none.
__host__ __device__ bool CheckRayObjectIntersection(staticGeom* geoms, int numberOfGeoms, ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, int& index)
{
	float closestT = 1000000.0, t;
	int closestIndex = -1;
	bool check = false;
	glm::vec3 selectedIntersectionPoint(0.0, 0.0, 0.0);
	glm::vec3 selectedNormal(0.0, 0.0, 0.0);
	for(int i = 0; i < numberOfGeoms; i++)
	{
		t = -1.0f;
		if(geoms[i].type == SPHERE)
		{
			t = sphereIntersectionTest(geoms[i], r, selectedIntersectionPoint, selectedNormal);
		}
		else if (geoms[i].type == CUBE)
		{
			t = boxIntersectionTest(geoms[i], r, selectedIntersectionPoint, selectedNormal);
		}
		else if (geoms[i].type == MESH)
		{
			//printf("Check: %f \t %f \t %f \n", geoms[i].triangles[0].vertices[0].x, geoms[i].triangles[0].vertices[0].y, geoms[i].triangles[0].vertices[0].z);
			//printf("i \t %d\n", i);
			t = MeshIntersectionTest(geoms[i], r, selectedIntersectionPoint, selectedNormal);
		}

		if(t >= 0 && t < closestT)
		{
			closestT = t - 0.001;
			closestIndex = i;
			intersectionPoint = selectedIntersectionPoint;
			normal = selectedNormal;
			check = true;
		}
	}
	index = closestIndex;
	return check;
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
	staticGeom* geoms, int numberOfGeoms, material* cudaMaterials, int numberOfMaterials, ray* InitialRays, int numOfRays)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	int BounceType = -1;
	glm::vec3 FinalColor(0.0f, 0.0f, 0.0f);
	
	if(InitialRays[index].keep == 1 && index < numOfRays)
	{
		ray r = InitialRays[index];
		glm::vec3 intersectionPoint(0.0f, 0.0f, 0.0f);
		glm::vec3 normal(0.0f, 0.0f, 0.0f);
		int closestIndex = -1;
		
		bool check = CheckRayObjectIntersection(geoms, numberOfGeoms, r, intersectionPoint, normal, closestIndex);

		glm::vec3 emittedColor;
		glm::vec3 unabsorbedColor;
		AbsorptionAndScatteringProperties AASP;
		r.keep = 0;
		material CM; 
		if(check && cudaMaterials[geoms[closestIndex].materialid].emittance > 0.005f) //If object is light then return light color.
		{
			CM = cudaMaterials[geoms[closestIndex].materialid]; //Storing the Material of the selected object
			FinalColor = CM.color * CM.emittance;
			r.keep = 0;
			BounceType = 0;
		}
		else if(check) //Do calculation for color
		{
			CM = cudaMaterials[geoms[closestIndex].materialid]; //Storing the Material of the selected object
			
			thrust::default_random_engine rng(hash(index*time*rayDepth));
			thrust::uniform_real_distribution<float> X1(0, 1);
			float u = X1(rng);
			float v = X1(rng);
			float w = X1(rng);
			thrust::uniform_real_distribution<float> X2(0, 5999999);
			int t = X2(rng);

			BounceType = calculateBSDF(r, geoms[closestIndex], closestIndex, intersectionPoint, normal, emittedColor, AASP, FinalColor, unabsorbedColor, CM, cudaMaterials, u, v, w, t);
		}
		if(BounceType == 0 || check == false)
			colors[r.pixelIndex] *= FinalColor;
		InitialRays[index] = r;

		if(rayDepth + 1 > MAX_DEPTH && InitialRays[index].keep == 1)
		{
			colors[r.pixelIndex] = glm::vec3(0.0f, 0.0f, 0.0f);
			InitialRays[index].keep == 0;
		}
	}
}

//TODO: FINISH THIS FUNCTION - Worked on this - Added Materials Data Pasing - ZM
//All Structures used are in sceneStructs.h
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management

void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, bool &changed, float FocalDistance){
  
	int traceDepth = 1; //determines how many bounces the raytracer traces
	
	// set up crucial magic
	int tileSize = 4;
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
		newStaticGeom.numOfTriangles = geoms[i].numOfTriangles;
		if(newStaticGeom.numOfTriangles > 0)
		{
			//newStaticGeom.triangles = new TriangleStruct[newStaticGeom.numOfTriangles];
			//for(int j = 0; j < newStaticGeom.numOfTriangles; j++)
				//newStaticGeom.triangles[j] = geoms[i].triangles[j];
			
			TriangleStruct* TriangleList = new TriangleStruct[newStaticGeom.numOfTriangles];
			for(int j = 0; j < newStaticGeom.numOfTriangles; j++)
			{
				TriangleList[j].index = geoms[i].triangles[j].index;
				for(int k = 0; k < 3; k++)
					TriangleList[j].vertices[k] = geoms[i].triangles[j].vertices[k];
				TriangleList[j].normal = geoms[i].triangles[j].normal;
			}
			TriangleStruct* triList = NULL;
			cudaMalloc((void**)&triList, newStaticGeom.numOfTriangles * sizeof(TriangleStruct));
			cudaMemcpy(triList, TriangleList, newStaticGeom.numOfTriangles * sizeof(TriangleStruct), cudaMemcpyHostToDevice);

			newStaticGeom.triangles = &triList[0];

		}
		geomList[i] = newStaticGeom;
	}
  
	//cudaMemcpy Sytax: cudaMalloc(Identifier, Sizeof)
	//cudaMemcpy Sytax: cudaMemcpy(ArrayOnDestinationDevice, ArrayOnSourceDevice, SizeofArray, D2H or H2D)

	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
	
	//package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;
  
	glm::vec3* InitVecs = new glm::vec3[3];
	glm::vec3 Avec = glm::cross(cam.view, cam.up);	//Avec
	glm::vec3 Bvec = glm::cross(Avec, cam.view);	//Bvec
	InitVecs[0] = cam.position + cam.view;	//Mvec
	InitVecs[1] = (Avec * (float)(cam.view.length() * tan(glm::radians(cam.fov.x)))) / (float) Avec.length();	//Hvec
	InitVecs[2] = (Bvec * (float)(cam.view.length() * tan(glm::radians(cam.fov.y)))) / (float) Bvec.length();	//Vvec

	glm::vec3* InitCamVecs = NULL;
	cudaMalloc((void**)&InitCamVecs, 3*sizeof(glm::vec3));
	cudaMemcpy( InitCamVecs, InitVecs, 3*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	material* cudaMaterials = NULL;
	cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudaMaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  
	int numOfRays = (renderCam->resolution.x * renderCam->resolution.y);
	ray* InitialRays = NULL;
	cudaMalloc((void**)&InitialRays,  numOfRays * sizeof(ray));
	
	///////////////////////////////////////////////
	clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage);
	//kernel launches

	GetRayCastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, InitCamVecs, (float)iterations, InitialRays, FocalDistance);
	//GetRayCastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, (float)iterations, InitialRays);

	dim3 fullBlocksPerGridSC = fullBlocksPerGrid;
	glm::vec2 Reso = renderCam->resolution;
	thrust::device_ptr<ray> InitRays(InitialRays);
	//Do first ray pass and store it
	while(traceDepth <= MAX_DEPTH && numOfRays > 0)
	{
		//do Ray tracing
		raytraceRay<<<fullBlocksPerGridSC, threadsPerBlock>>>(Reso, (float)iterations, cam, traceDepth,	cudaimage, cudageoms, numberOfGeoms, cudaMaterials, numberOfMaterials, InitialRays, numOfRays);
		
		traceDepth++;

		if(StreamCompactDepth > 0)
		{
			if(traceDepth % StreamCompactDepth == 0)
			{
				InitRays = thrust::device_ptr<ray>(InitialRays);
				thrust::device_ptr<ray> LastPtr = thrust::remove_if(InitRays, InitRays + numOfRays, is_alive());

				numOfRays = LastPtr.get() - InitRays.get();
				Reso.y = ceil(numOfRays / Reso.x); 
				fullBlocksPerGridSC = dim3((int)ceil(float(Reso.x)/float(tileSize)), (int)ceil(float(Reso.y)/float(tileSize)));
			}
		}
	}


	glm::vec3* camImage = NULL;
	cudaMalloc((void**)&camImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( camImage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	AccumalateColor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage, camImage, (float)iterations);

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	//free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudaMaterials );
	cudaFree( InitialRays );
	cudaFree( InitCamVecs );
	cudaFree( camImage );
	delete [] geomList;
	delete [] InitVecs;
	//delete renderCam->image;

	// make certain the kernel has completed 
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}