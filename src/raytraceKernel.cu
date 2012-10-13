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
#include "cuda_runtime.h"
#include <thrust\copy.h>

int * numThreads;

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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){

	int index = x + (y * resolution.x);

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
	glm::vec3 H = (A*float(CD*tan(fovx*(PI/180))))/float(glm::length(A));
	glm::vec3 V = (B*float(CD*tan(-fovy*(PI/180))))/float(glm::length(B));

	float sx = (x+(float)u01(rng))/(resolution.x);
	float sy = (y+(float)u01(rng))/(resolution.y);

	glm::vec3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
	glm::vec3 PmE = P-E;
	glm::vec3 R = E + (float(200)*(PmE))/float(glm::length(PmE));

	glm::vec3 direction = glm::normalize(R);
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0/iterations;
      color.y = image[index].y*255.0/iterations;
      color.z = image[index].z*255.0/iterations;

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

__device__ void rayCast( ray r, staticGeom* geoms, int numberOfGeoms, bounce& b, int& obj, float seed )
{
	float MAX_DEPTH = 100000000000000000;
	float depth = MAX_DEPTH;
	
	obj = -1;

	thrust::default_random_engine rng(hash(seed));
		thrust::uniform_real_distribution<float> u01(0,1);

	glm::vec3 color( 0 );
	glm::vec3 intersectionPoint( 0 );
	glm::vec3 intersectionNormal( 0 );

	for(int i=0; i<numberOfGeoms; i++)
	{
		if(geoms[i].type==SPHERE)
		{
			depth = sphereIntersectionTest(geoms[i], r, intersectionPoint, intersectionNormal);
		}else if(geoms[i].type==CUBE)
		{
			depth = boxIntersectionTest(geoms[i], r, intersectionPoint, intersectionNormal);
		}
		else if(geoms[i].type==PORTAL)
		{
			depth = portalIntersectionTest(geoms[i], geoms[geoms[i].materialid], r, intersectionPoint, intersectionNormal, (float)u01(rng) );
		}
		else
		{
			//lol?
		}
		if(depth<MAX_DEPTH && depth>-EPSILON)
		{
			obj = i;
			MAX_DEPTH = depth;
			b.matid = geoms[i].materialid;
			b.geoid = i;
			b.normal = intersectionNormal;
			b.position = intersectionPoint;
		}
	}

	b.incomingVector = r.direction;
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, bounce* bounces, int* numThreads)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if( index == 0 ) *numThreads = resolution.x*resolution.y;

	if((x<=resolution.x && y<=resolution.y))
	{
		int object;
		ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
		rayCast( r, geoms, numberOfGeoms, bounces[index], object, time*index );

		bounces[index].maxBounce = rayDepth;
		bounces[index].pixel = index;
		
		if( object == -1 )
		{
			bounces[index].color = glm::vec3( 0 );
			bounces[index].count = rayDepth;
			return;
		}
		
		bounces[index].color = glm::vec3( 1.0f ); //materials[geoms[object].materialid].color;
		if( geoms[object].type != PORTAL )
			bounces[index].count = ( materials[geoms[object].materialid].emittance > 0 ? rayDepth : 0 );
	}
}

//naive stream compression with double buffering
__global__ void compress(float its, int rayDepth, glm::vec3* colors, bounce* bounces, int* numThreads)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idx;

	if( index < *numThreads )
	{
		int sum = 0;
		if( bounces[index].count >= rayDepth )
		{
			idx = bounces[index].pixel;
			colors[idx] += bounces[idx].color;
		}
	}
}

__global__ void bounceAround(float time, cameraData cam, int rayDepth, glm::vec3* colors, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, bounce* bounces, int* numThreads)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int reflection;

	if( index < *numThreads )
	{
		if( bounces[index].count >= rayDepth )
			return;

		int timestamp = (int)time * index * (bounces[index].count+1);

		thrust::default_random_engine rng(hash(timestamp));
		thrust::uniform_real_distribution<float> u01(0,1);
		ray r;
		float x0 = (float)u01(rng);
		float x1 = (float)u01(rng);
		if(geoms[bounces[index].geoid].type==PORTAL)
		{
			r.direction = bounces[index].normal;
			reflection = 1;
		}
		else if( (float)u01(rng) < 0.5 || materials[bounces[index].matid].indexOfRefraction < EPSILON )
		{
			r.direction = calculateRandomDirectionInHemisphere( bounces[index].normal, x0, x1 );
			reflection = 0;
		}
		else
		{
			reflection = 1;
			r.direction = bounces[index].incomingVector - bounces[index].normal * ( 2 * glm::dot( bounces[index].incomingVector, bounces[index].normal ) ); 
		}
		r.direction = glm::normalize( r.direction );
		r.origin = bounces[index].position;
		
		int object;
		material mat = materials[bounces[index].matid];
		staticGeom geo = geoms[bounces[index].geoid];

		rayCast( r, geoms, numberOfGeoms, bounces[index], object, timestamp );
		
		if( object == -1 )
		{
			bounces[index].color = glm::vec3( 0 );
			bounces[index].count = rayDepth;
			return;
		}
		
		if( geo.type != PORTAL )
			bounces[index].color *= ( mat.emittance > 0 ? mat.emittance : 1.0f );
		
		if( !reflection )
		{
			bounces[index].color *= mat.color;
		}

		if( geo.type != PORTAL )
			bounces[index].count = ( mat.emittance > 0 ? rayDepth : bounces[index].count+1 );
		
		if( bounces[index].count >= rayDepth && mat.emittance <= EPSILON )
			bounces[index].color *= 0;
	}
}

struct is_done
{
	__host__ __device__
	bool operator()( const bounce x )
	{
		return x.count < x.maxBounce;
	}
};

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, int traceDepth){
  
	//determines how many bounces the raytracer traces

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
	dim3 threadsPerBounceBlock(tileSize*tileSize);
	dim3 fullBounceBlocksPerGrid((int)ceil(float(renderCam->resolution.x*renderCam->resolution.y)/float(tileSize*tileSize)));
  
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
		newStaticGeom.julia = geoms[i].julia;
		geomList[i] = newStaticGeom;
	}
  
	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
	material* cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
	
	bounce* cudabounces = NULL;
	bounce* Abounces = new bounce[ (int) (renderCam->resolution.x * renderCam->resolution.y) ];
	bounce* Bbounces = new bounce[ (int) (renderCam->resolution.x * renderCam->resolution.y) ];
	cudaMalloc( ( void ** ) &cudabounces, renderCam->resolution.x * renderCam->resolution.y * sizeof( bounce ) );
	
	static bounce* firstbounces = NULL;
	if( firstbounces == NULL ) cudaMalloc( ( void ** ) &firstbounces, renderCam->resolution.x * renderCam->resolution.y * sizeof( bounce ) );
	
	checkCUDAError("1111111111111!");
	//package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;

	//kernel launches
	if( iterations == 1 )
	{
		clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage);
		cudaMalloc((void**)&(numThreads), sizeof(int));
	}
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, 
														numberOfGeoms, cudamaterials, numberOfMaterials, firstbounces, numThreads);
	
	checkCUDAError("2222222222222!");
	cudaMemcpy( cudabounces, firstbounces, renderCam->resolution.x*renderCam->resolution.y*sizeof(bounce), cudaMemcpyDeviceToDevice);
	
	checkCUDAError("3333333333333!");
	
	int numT = 1;

	for( int i = 0; i < traceDepth && numT > 0; i++ )
	{
		bounceAround<<<fullBounceBlocksPerGrid, threadsPerBounceBlock>>>((float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials, cudabounces, numThreads);
		compress<<<fullBounceBlocksPerGrid, threadsPerBounceBlock>>>((float)iterations, traceDepth, cudaimage, cudabounces, numThreads);
		cudaMemcpy( &numT, numThreads, sizeof( int ), cudaMemcpyDeviceToHost );
		cudaMemcpy( Abounces, cudabounces, numT * sizeof( bounce ), cudaMemcpyDeviceToHost );
		std::cout << numT << " " << Bbounces << " " << std::endl;
		bounce* qbounce = thrust::copy_if( Abounces, Abounces + numT, Bbounces, is_done() );
		numT = qbounce - Bbounces;
		std::cout << numT << " " << Bbounces << " " << qbounce << std::endl;
		system("pause");
		cudaMemcpy( numThreads, &numT, sizeof( int ), cudaMemcpyHostToDevice );
		cudaMemcpy( cudabounces, Bbounces, numT * sizeof( bounce ), cudaMemcpyHostToDevice );
	}

	delete[] Abounces;
	delete[] Bbounces;
	
	checkCUDAError("In the Loop!");
	//compress<<<fullBounceBlocksPerGrid, threadsPerBounceBlock>>>((float)iterations, traceDepth, cudaimage, cudabounces, numThreads);
	
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);
	
	checkCUDAError("4444444444444!");
	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	
	//free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudabounces );
	cudaFree( cudamaterials );
	delete [] geomList;

	// make certain the kernel has completed 
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
	//system("pause");
}
