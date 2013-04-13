// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// This file contains starter code from Karl Yining and code from Gundeep Singh

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
//#include "obj.h:
//#include "glm/glm.hpp"

float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
triangle* primitives;

// used for stream compaction.
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

//Its cook torrance time
__host__ __device__ float cooktorrance(int bounces,int iterations,ray inray, ray& outray, glm::vec3 intersect, glm::vec3 N,
	glm::vec3& emittedcolor, glm::vec3& temp_color, material mat, float xi1, float xi2)
{
	//refered to pixar's link
	//See norm badler's notes for more information on microfacets and cook torrance.
	float gaussconstant=100;
	float roughness=0.04;
	glm::vec3 specularcolor= glm::vec3(1);

	float n1,n2;

	// things we need
	//normalized normal and vector to eye

	glm::vec3 Nn=glm::normalize(N);
	glm::vec3 Vn= glm::normalize(-1.0f*inray.direction);

	//float F, ktransmit;
	float m = roughness;
	n1=1.0f;
	n2= mat.indexOfRefraction;

	Fresnel fresnel=calculateFresnel( Nn, glm::normalize(inray.direction) , n1, n2);
	outray.direction= calculateRandomDirectionInHemisphere(N, xi1,xi2);
	outray.origin= intersect;
	glm::vec3 L =outray.direction;

	//glm::vec3 cook=glm::vec3(0);
	float n_dot_v= glm::dot(Nn, Vn);

	glm::vec3 Ln= glm::normalize(L);

	//half angle
	glm::vec3 H= glm::normalize(Vn+Ln);

	//float cook= pow(glm::dot(Nn,H),mat.specularExponent);

	float n_dot_h= glm::dot(Nn, H);
	float n_dot_l= glm::dot(Nn, Ln);
	float v_dot_h= glm::dot(Vn, H);

	float D;
	float alpha=glm::acos(n_dot_h);

	//micrtofacet distribution
	D= gaussconstant* exp(-(alpha*alpha)/(m*m));

	// geomteric attenuation factor
	float G= glm::min(1.0f,glm::min((2.0f*(n_dot_h*n_dot_v)/(v_dot_h)), (2.0f*(n_dot_h*n_dot_l)/(v_dot_h)))); 
	
	// sum cntributions
	float cook= ((fresnel.reflectionCoefficient*D*G)/(PI*n_dot_v)) *mat.specularColor.x;
	return ((cook));
	
}

void pathtracerReset()
{
	//~cudaimage();
}

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash1(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}
__host__ __device__ glm::vec3 generateRandomNumberFromThread2(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash1(index*time));
  thrust::uniform_real_distribution<float> u01(-0.5,0.5);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Kernel that does the initial raycast from the camera and caches the result. "First bounce cache, second bounce thrash!"
//host and device mean compile one version on the CPU and one version on CPU
__global__ void raycastFromCameraKernel(glm::vec2 resolution, float time, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, ray* ray_bundle, glm::vec3* temp, glm::vec3 *transmittedcolor, int iterations){
   
  float x = (blockIdx.x * blockDim.x) + threadIdx.x;
  float y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
  int index = x + (y * resolution.x);
  glm::vec3 rand=generateRandomNumberFromThread2(resolution,time,x,y);
   
  x=(float)x+((float)rand.x);
  y=(float)y+((float)rand.y);

  //standard camera raycast stuff from ray tracer
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

  ///////////////////////////////////
  /////////Depth Of Field ///////////
  ///////////////////////////////////

  // decide upon a focal plane distance
  float focalPlaneDist = 9.0f;
  glm::vec3 focusPoint;
  glm::vec3 rayDirection = direction;


  // to find the focus point start from eye in ray direction*focalplanedist
  focusPoint.x = eye.x + rayDirection.x * focalPlaneDist;
  focusPoint.y = eye.y + rayDirection.y * focalPlaneDist;
  focusPoint.z = eye.z + rayDirection.z * focalPlaneDist;

  // focallength
  float focalLength = 8.0f;
  // f-Number = focalLength/diameter
  float lensRadius = (focalLength/8.0)/2.0f;
  thrust::default_random_engine rng(hash1(index*iterations*123));
  thrust::uniform_real_distribution<float> u03(-lensRadius, lensRadius);
  glm::vec3 rayPointOnCamera = glm::vec3(eye.x + (float)u03(rng), eye.y + (float)u03(rng), eye.z);
		
  ray r;
  r.origin = rayPointOnCamera;
  r.direction = glm::normalize(focusPoint - rayPointOnCamera);

  ray_bundle[index].direction=r.direction;
  ray_bundle[index].origin=r.origin;
  ray_bundle[index].index_ray=index;
  ray_bundle[index].useful = true;

  // Reset temp
  transmittedcolor[index] = glm::vec3(1.0f,1.0f,1.0f);
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
// global means launched by CPU and run on device
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float time, ray* raybundle){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y)
  {
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

//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int iterations, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials,
							glm::vec3* renderimage, ray* raybundle, int bounces, glm::vec3* temp, obj* cudameshes,
							int numberofmeshes, float *device_vbo, float vbosize, triangle* faces, float* device_nbo,
							int nbosize, glm::vec3 *tranmitted_color)
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
		for(int i=0; i<=numberOfGeoms; i++)
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

			else if( geoms[i].type==MESH)
			{
				//printf("okay in mesh");
				for ( int j=0; j<vbosize/9; ++j)
				{
				//	 
					glm::vec3 temp_intersectionPoint;
					glm::vec3 temp_normal;
					
					depth = RayTriangleIntersect(geoms[i], raybundle[index], faces[j].p0,faces[j].p1,faces[j].p2,
						faces[j].n0,intersectionPoint, temp_normal);
					if (depth>-EPSILON)
					{
						if (tmin<=EPSILON || depth<tmin+EPSILON)
						{
							tmin=depth;
							POI=intersectionPoint;
							normal=temp_normal;//intersectionNormal;
							Object_number=i;
						}
					}
				}
			}
		}
				
		if(tmin<EPSILON)  // ray does not hit anything
		{
			raybundle[index].useful=false;
			temp[raybundle[index].index_ray]= glm::vec3(0,0,0);
			colors[raybundle[index].index_ray]=(old_color)/time; //old_color+
		}
		else  // ray hits something
		{
			// ITS A GOOD TIME TO THINK ABOUT SUBSURFCAE SCATERRING

			 if (materials[geoms[Object_number].materialid].hasScatter>0 || glm::length(materials[geoms[Object_number].materialid].absorptionCoefficient) >0)
			{

				AbsorptionAndScatteringProperties something;
				something.absorptionCoefficient= materials[geoms[Object_number].materialid].absorptionCoefficient;
				something.reducedScatteringCoefficient= materials[geoms[Object_number].materialid].reducedScatterCoefficient;

				//random numbers
				thrust::default_random_engine rng(hash1(index*iterations));
				thrust::uniform_real_distribution<float> u01(0,1);
				float randomFloatForScatteringDistance=(float)u01(rng);
				float randomFloat2=(float)u01(rng);
				float randomFloat3=(float)u01(rng);
				
				
				// Find the scattering dist
				float scatteringDistance= -log(randomFloatForScatteringDistance)/something.reducedScatteringCoefficient;

				float find_t= (glm::length(POI - raybundle[index].origin));

				//check this to if the ray is still in the object
				if(scatteringDistance< find_t)
				{
					ray nextRay;
					raybundle[index].origin=POI;
					nextRay.origin= raybundle[index].origin + raybundle[index].direction*scatteringDistance;
					nextRay.direction = getRandomDirectionInSphere (randomFloat2,  randomFloat3);
					//nextRay.origin=POI;
					raybundle[index].direction= nextRay.direction;
					raybundle[index].origin= nextRay.origin;
					tranmitted_color[raybundle[index].index_ray]*= calculateTransmission(something.absorptionCoefficient, scatteringDistance);
						// TEST VALUES before and after absorption.
						//printf("color before absorption %f   %f    %f \n" , temp[raybundle[index].index_ray].x,temp[raybundle[index].index_ray].y,temp[raybundle[index].index_ray].z);
						//temp[raybundle[index].index_ray]= calculateTransmission(something.absorptionCoefficient, scatteringDistance);
						//printf("color after absorption %f   %f    %f \n" , temp[raybundle[index].index_ray].x,temp[raybundle[index].index_ray].y,temp[raybundle[index].index_ray].z);
					return; // to handle each bounce seperately
				}
				else
				// only absorption
				tranmitted_color[raybundle[index].index_ray]*= calculateTransmission(something.absorptionCoefficient,find_t);
				//return;
			}

			// when the ray hits a REFLECTIVE only material
			else if (materials[geoms[Object_number].materialid].hasReflective>0 
					&& materials[geoms[Object_number].materialid].hasRefractive==0)
			{
				glm::vec3 reflectedRay= calculateReflectionDirection(normal,raybundle[index].direction); 
				raybundle[index].direction=reflectedRay;
				raybundle[index].origin=POI;
				return; // this return means generate a new ray.
			}
			// when the ray hits a refractive material
			if (materials[geoms[Object_number].materialid].hasReflective>0 
				  && materials[geoms[Object_number].materialid].hasRefractive>0 )
			{
				float n1,n2;
				n1=1.0f;
				n2=materials[geoms[Object_number].materialid].indexOfRefraction;

				ray outRay;
				outRay.direction= calculateTransmissionDirection(normal,raybundle[index].direction, n1,n2);
				Fresnel fresnel = calculateFresnel(normal, raybundle[index].direction,n1,n2);

				//using russian roullete to find if reflection or refraction
				thrust::default_random_engine rng(hash1(index*bounces*time));
				thrust::uniform_real_distribution<float> u01(0,1);
				float xi1=(float)u01(rng);
				float xi2=(float)u01(rng);
				float xi3=(float)u01(rng);
				float russianRoulette = xi3;

				if ( russianRoulette < fresnel.transmissionCoefficient)
				{
					//do refraction
					outRay.direction= glm::normalize(calculateTransmissionDirection(normal, raybundle[index].direction, n1,n2));
					outRay.origin= POI;
					raybundle[index].origin=outRay.origin;
					raybundle[index].direction=outRay.direction;
					if (bounces==1)
					{
						temp[raybundle[index].index_ray]=materials[geoms[Object_number].materialid].color;
					}
					else //if (bounces==2 && bounces <9)
					{
						temp[raybundle[index].index_ray]= temp[raybundle[index].index_ray]*materials[geoms[Object_number].materialid].color; 
					}
				}
				else
				{
					// do refelction
					outRay.direction = glm::normalize(calculateReflectionDirection(normal,raybundle[index].direction));
					outRay.origin=POI;
					raybundle[index].origin=outRay.origin;
					raybundle[index].direction=outRay.direction;
					// since the ray got reflected it will not go inside the object so no more intersection tests.
					return;
				}

				// redo all the intersection tests ( THINK ABOUT IT THERE MUST A WAY TO AVOID THIS)
				for(int i=0; i<=numberOfGeoms; i++)
				{
					glm::vec3 intersectionPoint;
					glm::vec3 intersectionNormal;
					if(geoms[i].type==SPHERE)
					{
						depth = sphereIntersectionTest(geoms[i], outRay, intersectionPoint, intersectionNormal);
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
						depth = boxIntersectionTest(geoms[i], outRay, intersectionPoint, intersectionNormal);
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

					else if( geoms[i].type==MESH)
					{
						//printf(" In mesh");
						for ( int j=0; j<vbosize/9; ++j)
						{
						
							glm::vec3 temp_intersectionPoint;
							glm::vec3 temp_normal;
					
							depth = RayTriangleIntersect(geoms[i], outRay, faces[j].p0,faces[j].p1,faces[j].p2,
								faces[j].n0,intersectionPoint, temp_normal);
							if (depth>-EPSILON)
							{
								if (tmin<=EPSILON || depth<tmin+EPSILON)
								{
									tmin=depth;
									POI=intersectionPoint;
									normal=temp_normal;//intersectionNormal;
									Object_number=i;
								}
							}
						}
					}
				}
				
				// now the ray is inside and you have to find out the direction when it goes out
				n2=1.0f;
				n1=materials[geoms[Object_number].materialid].indexOfRefraction;
				outRay.direction=calculateTransmissionDirection(normal,outRay.direction,n1,n2);
				outRay.origin=POI;
				raybundle[index].origin=outRay.origin;
				raybundle[index].direction=outRay.direction;
			}

			else // diffuse surface
			{
		 		glm::vec3 random1_num=generateRandomNumberFromThread(resolution,time*(bounces),x,y);
				random_direction=calculateRandomDirectionInHemisphere(normal,random1_num.x,random1_num.y);
				random_direction=random_direction;
				ray outray;
				if (materials[geoms[Object_number].materialid].specularExponent>0)
				{
					//glm::vec3 view_dir= glm::normalize(cam.position-POI);
					//glm::vec3 reflectedlight= calculateReflectionDirection(normal, -raybundle[index].direction);

					float specterm= cooktorrance( bounces,iterations,raybundle[index], outray, POI,normal, temp[raybundle[index].index_ray],
					colors[raybundle[index].index_ray],materials[geoms[Object_number].materialid],
					random1_num.x,random1_num.y);
					temp[raybundle[index].index_ray]= temp[raybundle[index].index_ray]*materials[geoms[Object_number].materialid].color
						+1.0f*specterm;
					raybundle[index].direction=outray.direction;
					raybundle[index].origin=outray.origin;
					specterm=0;
				}
				else
				temp[raybundle[index].index_ray]= temp[raybundle[index].index_ray]*materials[geoms[Object_number].materialid].color;

				raybundle[index].direction=random_direction;
				raybundle[index].origin=POI;

				// when the ray hits light
				if (materials[geoms[Object_number].materialid].emittance>0 )
				 {
					colors[raybundle[index].index_ray]=(old_color+(tranmitted_color[raybundle[index].index_ray]* temp[raybundle[index].index_ray]*materials[geoms[Object_number].materialid].color*materials[geoms[Object_number].materialid].emittance))/time; //old_color+
					raybundle[index].useful=false;
					return;
				}
			}
		}
	}
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations,
					material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,
					float* vbo, float* nbo,float* cbo, int vbosize, int nbosize,int cbosize, obj* objs, int numberofmeshes,
					int number_of_faces, triangle* tri_faces, int* ibo, int ibosize){
  
  int bounces=0;
  int activerays=(int)renderCam->resolution.x*(int)renderCam->resolution.y;

  // assigning the threads and blocks
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
  
  glm::vec3* transmittedcolor=NULL;
  cudaMalloc((void**)&transmittedcolor,(int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));

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

	//// Hack for Motion Blur after every 8th iteration move the object just a tiny bit. Thanks to path tracing.
		if (i == 8) 
		{
			newStaticGeom.translation.x += 0.00005;
			geoms[i].translations[frame] = newStaticGeom.translation;
			glm::mat4 transform = utilityCore::buildTransformationMatrix(newStaticGeom.translation, newStaticGeom.rotation, newStaticGeom.scale);
			geoms[i].transforms[frame] =  utilityCore::glmMat4ToCudaMat4(transform);
			geoms[i].inverseTransforms[frame] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
			geoms[i].tranposeTranforms[frame] = utilityCore::glmMat4ToCudaMat4(glm::inverse(glm::transpose(transform)));
		}

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
  
  obj* cudaobjs=NULL;
  cudaMalloc((void**)&cudaobjs, numberofmeshes*sizeof(obj));
  cudaMemcpy(cudaobjs, objs, numberofmeshes*sizeof(obj), cudaMemcpyHostToDevice);  

  //------------------------------
  //MEMORY STUFF
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));
  cudaMemcpy(primitives,tri_faces, (ibosize/3)*sizeof(triangle),cudaMemcpyHostToDevice);

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_nbo =NULL;
  cudaMalloc ((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy(device_nbo, nbo, nbosize*sizeof(float),cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));

  //kernel call for camera rays
  raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>( renderCam->resolution, (float)iterations, cam.position, cam.view, cam.up, cam.fov, raypacket, temp_color, transmittedcolor,iterations);
  
  //raytrace kernel launches
  while(bounces<10)
  {
  		bounces++;
  		dim3 fullBlockPerGridSC((int)ceil(float(renderCam->resolution.x)/float(tileSize)), ((int)activerays)/(float(tileSize)*(int)ceil(float(renderCam->resolution.x))));
  		raytraceRay<<<fullBlockPerGridSC, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, iterations, 
  		cudaimage, cudageoms, numberOfGeoms, cudamaterials,numberOfMaterials,renderimage,
  		raypacket,bounces,temp_color, cudaobjs, numberofmeshes,device_vbo,vbosize,primitives, device_nbo, nbosize, transmittedcolor);

		// stream compaction using thrust 
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
  cudaFree(transmittedcolor);
  cudaFree(cudaobjs);
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( device_nbo );
  cudaFree(primitives);
  delete [] geomList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();
  checkCUDAError("Kernel failed!");
}
