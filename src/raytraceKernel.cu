// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com
#include <thrust/scan.h>
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
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>



//#define DEPTHOFFIELD
//#define SUPERSAMPLING
float Lensdistance=5;
int NumberOfSampling=5;


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
  thrust::uniform_real_distribution<float> u01(-1,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

__global__ void calculateRaycastFromCameraKernel(cameraData cam, float time,ray* rayArray){

	
	ray r;
	r.origin = cam.position;
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	thrust::default_random_engine rng(hash((int(index)*time)));
	thrust::uniform_real_distribution<float> u01(0,3);
	thrust::uniform_real_distribution<float> u02(-1,1);
	//float noisex =((float)u01(rng))*0.5f;
	//float noisey =((float)u01(rng))*0.5f;
	float noisex ,noisey;
	float dt=1.0f/6.0f;
	float dt2=1.0/3.0f;
	float russianRoulette = (float)u01(rng);
	float random2=(float)u02(rng);
	if(russianRoulette<0.33f){
		noisex=-dt2+((float)u02(rng))*dt;
		noisey=dt2+((float)u02(rng))*dt;;
	}else if(russianRoulette<0.67f){
		noisex=((float)u02(rng))*dt;
		noisey=dt2+((float)u02(rng))*dt;
	}else if(russianRoulette<1.0f){
		noisex=dt2+((float)u02(rng))*dt;
		noisey=dt2+((float)u02(rng))*dt;
	}else if(russianRoulette<1.33f){
		noisex=-dt2+((float)u02(rng))*dt;
		noisey=((float)u02(rng))*dt;;
	}else if(russianRoulette<1.67f){
		noisex=((float)u02(rng))*dt;
		noisey=((float)u02(rng))*dt;
	}else if(russianRoulette<2.0f){
		noisex=dt2+((float)u02(rng))*dt;
		noisey=((float)u02(rng))*dt;
	}if(russianRoulette<2.33f){
		noisex=-dt2+((float)u02(rng))*dt;
		noisey=-dt2+((float)u02(rng))*dt;;
	}else if(russianRoulette<2.67f){
		noisex=((float)u02(rng))*dt;
		noisey=-dt2+((float)u02(rng))*dt;
	}else if(russianRoulette<3.0f){
		noisex=dt2+((float)u02(rng))*dt;
		noisey=-dt2+((float)u02(rng))*dt;
	}


	if((x<=cam.resolution.x )&&( y<=cam.resolution.y)){
    float y1=cam.resolution.y-y;
	float x1=cam.resolution.x-x;
	glm::vec3 A = glm::cross(cam.view,cam.up); //A= view^up
	float ALength=glm::length(A);
	glm::vec3 B =  glm::cross(A,cam.view);	//B <- A * C
	float BLength=glm::length(B);
    glm::vec3 M = cam.position + cam.view;	//M=E+C
	float viewLength=glm::length(cam.view);
	glm::vec3 H = A*viewLength * (float)tan(cam.fov.x*(PI/180.0f))/ ALength; //H <- (A|C|tan)/|A|
	glm::vec3 V = B*viewLength *(float)tan(cam.fov.y*(PI/180.0f)) / BLength;   // V <- (B|C|tan)/|B|  
	//glm::vec3 P=M+(2*((float)x1/(float)(cam.resolution.x-1))-1)*H+(2*((float)y1/(float)(cam.resolution.y-1))-1)*V;
	glm::vec3 P=M+(2*(float)(x1+noisex)/(float)(cam.resolution.x-1)-1)*H+(2*(float)(y1+noisey)/(float)(cam.resolution.y-1)-1)*V;
	glm::vec3 D=P-cam.position;
	r.direction=glm::normalize(D);
    rayArray[index]=r;
	}
	return;
}

__global__ void calculateRaycastFromCameraKernelSuperSampling(cameraData cam, float time,int sampleround,ray* rayArray){

	ray r;
	r.origin = cam.position;
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	thrust::default_random_engine rng(hash(sampleround*index*time));
	thrust::uniform_real_distribution<float> u01(0,1);
	thrust::uniform_real_distribution<float> u02(-1,1);
	//float noisex =((float)u01(rng))*0.45f;
	//float noisey =((float)u01(rng))*0.45f;
	float noisex ,noisey;
	float russianRoulette = (float)u01(rng);
	if(russianRoulette<0.25){
		noisex=0.25+((float)u02(rng))*0.125f;
		noisey=0.25+((float)u02(rng))*0.125f;;
	}else if(russianRoulette<0.5){
		noisex=-0.25+((float)u02(rng))*0.125f;
		noisey=0.25+((float)u02(rng))*0.125f;
	}else if(russianRoulette<0.75){
		noisex=0.25+((float)u02(rng))*0.125f;
		noisey=-0.25+((float)u02(rng))*0.125f;
	}else{
		noisex=-0.25+((float)u02(rng))*0.125f;
		noisey=-0.25+((float)u02(rng))*0.125f;
	}



	if((x<=cam.resolution.x )&&( y<=cam.resolution.y)){
    float y1=cam.resolution.y-y;
	float x1=cam.resolution.x-x;
	glm::vec3 A = glm::cross(cam.view,cam.up); //A= view^up
	float ALength=glm::length(A);
	glm::vec3 B =  glm::cross(A,cam.view);	//B <- A * C
	float BLength=glm::length(B);
    glm::vec3 M = cam.position + cam.view;	//M=E+C
	float viewLength=glm::length(cam.view);
	glm::vec3 H = A*viewLength * (float)tan(cam.fov.x*(PI/180.0f))/ ALength; //H <- (A|C|tan)/|A|
	glm::vec3 V = B*viewLength *(float)tan(cam.fov.y*(PI/180.0f)) / BLength;   // V <- (B|C|tan)/|B|  
	//glm::vec3 P=M+(2*((float)x1/(float)(cam.resolution.x-1))-1)*H+(2*((float)y1/(float)(cam.resolution.y-1))-1)*V;
	glm::vec3 P=M+(2*(float)(x1+noisex)/(float)(cam.resolution.x-1)-1)*H+(2*(float)(y1+noisey)/(float)(cam.resolution.y-1)-1)*V;
	glm::vec3 D=P-cam.position;
	r.direction=glm::normalize(D);
    rayArray[index]=r;
	}
	return;
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

__global__ void raytracefromCameraKernel(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int  numberOfGeoms,material* materials,ray* cudaFirstRays,rayData* rayList){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x>=resolution.x )||( y>=resolution.y))return;
  if(index>=(resolution.x*resolution.y))return;
  ray r=cudaFirstRays[index];
 
  glm::vec3 finalColor=glm::vec3(0.0f,0.0f,0.0f);   
	//find first intersection
	float distance=-1.0f;
	glm::vec3 interestPoint=glm::vec3(0,0,0);
	glm::vec3 normal=glm::vec3(0,0,0);
	int geoID=-1;
	for(int i=0; i<numberOfGeoms; i++){
		float tempdistance=-1.0;
		glm::vec3 tempInterestPoint=glm::vec3(0,0,0);
	    glm::vec3 tempNormal=glm::vec3(0,0,0);
		if(geoms[i].type==SPHERE){
			tempdistance=sphereIntersectionTest(geoms[i],r,tempInterestPoint,tempNormal);
		}else if(geoms[i].type==CUBE){
			tempdistance=boxIntersectionTest(geoms[i],r,tempInterestPoint,tempNormal);
		}

		if((abs(distance+1.0f)<1e-3)&&(tempdistance>0.001f)){
			distance=tempdistance;
			normal=tempNormal;
			interestPoint=tempInterestPoint;
			geoID=i;
		}else if((tempdistance>0.001f)&&(tempdistance<distance)){
			distance=tempdistance;
			normal=tempNormal;
			interestPoint=tempInterestPoint;
			geoID=i;
		}

		}

	rayData nextray;
	//can not find intersection ,ray ends
	if(geoID==-1){
		finalColor=glm::vec3(0,0,0);
		nextray.dirty=0;
	}else {
			material m=materials[geoms[geoID].materialid];
			if(m.emittance>0){      ///light source
			
				 finalColor=m.emittance*m.color;
				 nextray.dirty=0;
			}else{

					glm::vec3 emittedColor=glm::vec3(0.0f,0.0f,0.0f);
					glm::vec3 unabsorbedColor=glm::vec3(0.0f,0.0f,0.0f);
					
					AbsorptionAndScatteringProperties currentAbsorptionAndScattering;
					calculateBSDF(cam.position,r,interestPoint,normal,emittedColor, currentAbsorptionAndScattering,
						finalColor, unabsorbedColor, m, (float)index*time, nextray);
					nextray.dirty=1;
	
			 }
	}


	 colors[index] =finalColor;
	 nextray.x=x;
	 nextray.y=y;
	 rayList[index]=nextray;

}


__global__ void iterationRaytrace(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
	staticGeom* geoms, int numberOfGeoms,material* materials,rayData* rayList,int maxnum,int WIDTH,int currentDepth){
 
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int threadIndex=x+WIDTH*y;

  if((x>=WIDTH )||( y>=WIDTH))return;
  if(threadIndex>=maxnum) return;
  rayData rd=rayList[threadIndex];
  int colorIndex = rd.x + (rd.y * resolution.x);
  ray r=rd.newray;


  glm::vec3 finalColor=glm::vec3(0.0f,0.0f,0.0f);

	//find first intersection
	float distance=-1.0f;
	glm::vec3 interestPoint=glm::vec3(0,0,0);
	glm::vec3 normal=glm::vec3(0,0,0);
	int geoID=-1;
	for(int i=0; i<numberOfGeoms; i++){
		float tempdistance=-1.0;
		glm::vec3 tempInterestPoint=glm::vec3(0,0,0);
	    glm::vec3 tempNormal=glm::vec3(0,0,0);
		if(geoms[i].type==SPHERE){
			tempdistance=sphereIntersectionTest(geoms[i],r,tempInterestPoint,tempNormal);
		}else if(geoms[i].type==CUBE){
			tempdistance=boxIntersectionTest(geoms[i],r,tempInterestPoint,tempNormal);
		}

		if((abs(distance+1.0f)<1e-3)&&(tempdistance>0.001f)){
			distance=tempdistance;
			normal=tempNormal;
			interestPoint=tempInterestPoint;
			geoID=i;
		}else if((tempdistance>0.001f)&&(tempdistance<distance)){
			distance=tempdistance;
			normal=tempNormal;
			interestPoint=tempInterestPoint;
			geoID=i;
		}

		}

	rayData nextray;
	
	//can not find intersection ,ray ends
	 if(geoID==-1){
		finalColor=glm::vec3(0,0,0);
		nextray.dirty=0;
	}else {
			material m=materials[geoms[geoID].materialid];
			if(m.emittance>0){      ///light source
				finalColor=m.emittance*m.color;
				 nextray.dirty=0;
			}else{

					glm::vec3 emittedColor=glm::vec3(0.0f,0.0f,0.0f);
					glm::vec3 unabsorbedColor=glm::vec3(0.0f,0.0f,0.0f);
					
					AbsorptionAndScatteringProperties currentAbsorptionAndScattering;
					calculateBSDF(cam.position,r,interestPoint,normal,emittedColor, currentAbsorptionAndScattering,
						finalColor, unabsorbedColor, m, (float)threadIndex*time*currentDepth, nextray);
					nextray.dirty=1;
	
			 }
	}


	 glm::vec3 precolor=colors[colorIndex];
	 colors[colorIndex] = glm::vec3(precolor.x*finalColor.x,precolor.y*finalColor.y,precolor.z*finalColor.z);
	 nextray.x=rd.x;
	 nextray.y=rd.y;
	 rayList[threadIndex]=nextray;

}


 __global__ void mergeImage(glm::vec2 resolution,glm::vec3* previousColors,glm::vec3* currentColors,float time){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
   if((x<=resolution.x) && (y<=resolution.y)){
	    glm::vec3 currentColor=currentColors[index];
		 glm::vec3 previousColor=previousColors[index];

	   currentColors[index]=currentColor/time+previousColor*(time-1.0f)/time;
   }

 }

 __global__ void scanRound( glm::vec2 dim,int* previewArray,int* nextarray, int boundary,int flag){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * dim.x);

	if((x>=dim.x)||(y>=dim.y))return;
	if(index<boundary){
	if(index>=flag)
			nextarray[index]=previewArray[index]+previewArray[index-flag];
		else
			nextarray[index]=previewArray[index];
	}
 }

 
 __global__ void copyBack( glm::vec2 dim,int* previewArray,int* nextarray, int boundary){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * dim.x);
	if((x>=dim.x)||(y>=dim.y))return;
	if(index<boundary){
	previewArray[index]=nextarray[index];
	}
 }

 __global__ void getValue(glm::vec2 dim, rayData* raydatalist,int* indexlist,int boundary){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * dim.x);

	if((x>=dim.x)||(y>=dim.y))return;
	if(index<boundary) {
	indexlist[index]=raydatalist[index].dirty;
	}
	
}

__global__ void inclusiveTOexclusive(glm::vec2 dim, rayData* raydatalist,int* indexlist,int boundary){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * dim.x);
	if((x>=dim.x)||(y>=dim.y))return;
	if(index<boundary) {
	indexlist[index]-=raydatalist[index].dirty;
	}
	
}


void scan(dim3 DimBlock, dim3 DimThread, glm::vec2 dim,rayData* rayList,int* indexArray,int* tempArray, int boundary){

 
 getValue<<<DimBlock,DimThread>>>(dim,rayList,indexArray,boundary);
 checkCUDAError("Kernel failed! scan-1");
  for(int d=1;d<=ceil((float)log2((float)boundary));d++){  
		int flag=(int)pow(2.0f,d-1);
		//std::cout<<"d="<<d<<std::endl;
		//std::cout<<"flag="<<flag<<std::endl;
	    scanRound<<<DimBlock,DimThread>>>(dim,indexArray,tempArray,boundary,flag);
		checkCUDAError("Kernel failed! scan-2-1");
		copyBack<<<DimBlock,DimThread>>>(dim,indexArray,tempArray,boundary);
		// cudaMemcpy( indexArray, tempArray, (boundary)*sizeof(int), cudaMemcpyDeviceToDevice);
		checkCUDAError("Kernel failed! scan-2-2");
 }	

 inclusiveTOexclusive<<<DimBlock,DimThread>>>(dim,rayList,indexArray,boundary);
 checkCUDAError("Kernel failed! scan-3");

 }



__global__ void stringcompaction(glm::vec2 dim,rayData* rayList,rayData *newdataArray,int* indexArray,int maxBoundary){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * dim.x);
 if((x>=dim.x)||(y>=dim.y))return;
  if(index<maxBoundary){
	 if(rayList[index].dirty){
	  newdataArray[indexArray[index]]=rayList[index];
	 }
	}
 }
  
//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, ray* firstRays){
  
  int traceDepth =100; //determines how many bounces the raytracer traces

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
  
   //--------------------------------
  //package materials
  material* materialList=new material[numberOfMaterials];

  for(int i=0;i<numberOfMaterials;i++){
	  material newmaterial;
	  newmaterial.color=materials[i].color;
	  newmaterial.specularExponent=materials[i].specularExponent;
	  newmaterial.specularColor=materials[i].specularColor;
	  newmaterial.hasReflective=materials[i].hasReflective;
	  newmaterial.hasRefractive=materials[i].hasRefractive;
	  newmaterial.indexOfRefraction=materials[i].indexOfRefraction;
	  newmaterial.hasScatter=materials[i].hasScatter;
	  newmaterial.absorptionCoefficient=materials[i].absorptionCoefficient;
	  newmaterial.reducedScatterCoefficient=materials[i].reducedScatterCoefficient;
	  newmaterial.emittance=materials[i].emittance;

	  materialList[i]=newmaterial;
  }

  material* cudamatrials=NULL;
  cudaMalloc((void**)&cudamatrials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamatrials, materialList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
	 	  
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;


   //first Rays cudaMemory Pointer
  ray* cudaFirstRays = NULL;
  cudaMalloc((void**)&cudaFirstRays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));

  //saved first ray Color cudaMemory Pointer
  rayData* cudaRayList = NULL;
  cudaMalloc((void**)&cudaRayList, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(rayData));

  //scan result cudaMemory Pointer
  int* scanResultRayList = NULL;
  cudaMalloc((void**)&scanResultRayList, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(int));
  
  //for string compaction
  int *tempscanResultRayList=NULL;
  cudaMalloc((void**)&tempscanResultRayList, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(int));
  rayData* newRayDataList=NULL;
  cudaMalloc((void**)&newRayDataList, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(rayData));
#ifdef SUPERSAMPLING

  
   #ifdef DEPTHOFFIELD
thrust::default_random_engine rng(hash((float)iterations));
    thrust::uniform_real_distribution<float> u01(-1,1);
	float xDist=(float)u01(rng);
	float yDist=(float)u01(rng);

	float length=abs(glm::dot(glm::vec3(glm::vec3(0,0,0)-cam.position),cam.view));
	glm::vec3 focalPos=cam.position+cam.view*length;
	cam.position+=100.0f*glm::vec3(xDist*Lensdistance*1/cam.resolution.x,yDist*Lensdistance*1/cam.resolution.y,0.0f);
	cam.view=glm::normalize(focalPos-cam.position);
   #endif

	glm::vec3* cudaimageSuperSamping = NULL;
	cudaMalloc((void**)&cudaimageSuperSamping, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	
	int SampleCount=1;
	while(SampleCount<=NumberOfSampling){

#endif
	// save the first Ray Direction
#ifndef SUPERSAMPLING
    calculateRaycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(cam,(float)iterations,cudaFirstRays);
	//cudaMemcpy(firstRays, cudaFirstRays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray), cudaMemcpyDeviceToHost);
#else
	calculateRaycastFromCameraKernelSuperSampling<<<fullBlocksPerGrid, threadsPerBlock>>>(cam,(float)iterations,SampleCount,cudaFirstRays);
#endif
   //calculate first ray color

  raytracefromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,cudamatrials,cudaFirstRays,cudaRayList);

  checkCUDAError("Kernel failed! 2");

  
  //scan
   int maxBoundary=(int)renderCam->resolution.x*(int)renderCam->resolution.y;
   scan(fullBlocksPerGrid, threadsPerBlock,renderCam->resolution,cudaRayList,scanResultRayList,tempscanResultRayList,maxBoundary);
   checkCUDAError("Kernel failed! 3");

  int flag=-1;
  cudaMemcpy(&flag,&(cudaRayList[maxBoundary-1].dirty),sizeof(int),cudaMemcpyDeviceToHost);

  // rays for string compaction
  stringcompaction<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,cudaRayList,newRayDataList,scanResultRayList,maxBoundary);
  cudaMemcpy(cudaRayList,newRayDataList,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(rayData), cudaMemcpyDeviceToDevice);

  //get number of rays in raylist
  int numberOfRays=0;
  cudaMemcpy(&numberOfRays,&(scanResultRayList[maxBoundary-1]),sizeof(int),cudaMemcpyDeviceToHost);
   if(flag==1)
	   numberOfRays++;
  checkCUDAError("Kernel failed! list-newlist");

  //iteration kernel launches
  int currDepth=1;
  while((numberOfRays>0)&&(currDepth<=traceDepth)){ 
	 //std::cout<<"depth="<<currDepth<<std::endl;
	int length=ceil(sqrt((float)(numberOfRays)));
    dim3 newthreadsPerBlock(tileSize, tileSize);
	dim3 newfullBlocksPerGrid((int)ceil(float(length)/float(tileSize)), (int)ceil(float(length)/float(tileSize)));
	glm::vec2 dim=glm::vec2(length,length);

	iterationRaytrace<<<newfullBlocksPerGrid,newthreadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,cudamatrials,cudaRayList,numberOfRays,length,currDepth);
	checkCUDAError("Kernel failed! 4-0");

	//get flag
	cudaMemcpy(&flag,&(cudaRayList[numberOfRays-1].dirty),sizeof(int),cudaMemcpyDeviceToHost);
	checkCUDAError("Kernel failed! 4-1");

	scan(newfullBlocksPerGrid,newthreadsPerBlock,dim,cudaRayList,scanResultRayList,tempscanResultRayList,numberOfRays);
	checkCUDAError("Kernel failed! 4-2");
	
	//string compaction
	stringcompaction<<<newfullBlocksPerGrid,newthreadsPerBlock>>>(dim,cudaRayList,newRayDataList,scanResultRayList,numberOfRays);
	cudaMemcpy(cudaRayList,newRayDataList,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(rayData), cudaMemcpyDeviceToDevice);

	//get number of rays
	cudaMemcpy(&numberOfRays,&scanResultRayList[numberOfRays-1],sizeof(int),cudaMemcpyDeviceToHost);
	if(flag==1)
	   numberOfRays++;

	checkCUDAError("Kernel failed! 4-3");
	//std::cout<<"number of rays"<<numberOfRays<<std::endl; 
	currDepth++;
	
  }

 #ifdef SUPERSAMPLING
  if(SampleCount!=1)
	mergeImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,cudaimageSuperSamping,cudaimage,(float)SampleCount);
	checkCUDAError("Kernel failed!supersamping-1");

	cudaMemcpy( cudaimageSuperSamping, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    SampleCount++;
	}
#endif

  //combine several iteration together

  // previous image cudaMemory Pointer
  glm::vec3* previousImage = NULL;
  cudaMalloc((void**)&previousImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( previousImage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  checkCUDAError("Kernel failed! 6");

  mergeImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,previousImage,cudaimage,(float)iterations),
  checkCUDAError("Kernel failed! 7");
  
  //retrieve image from GPU
  cudaMemcpy(renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);


   sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);
  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree(cudaFirstRays);
  cudaFree(cudaRayList);
  cudaFree(scanResultRayList);
  cudaFree( cudageoms );
  cudaFree(cudamatrials);
  cudaFree(previousImage);
  cudaFree(tempscanResultRayList);
  cudaFree(newRayDataList);
#ifdef SUPERSAMPLING
  	cudaFree(cudaimageSuperSamping);
#endif
  delete geomList;
  delete materialList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
