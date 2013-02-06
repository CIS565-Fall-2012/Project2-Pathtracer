// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
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


void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
}

struct IsRayAbsent
{
	__host__ __device__
	bool operator()(const RayPackage& rayPackage)
	{
		return !rayPackage.isPresent;
	}
};

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

////Kernel that does the initial raycast from the camera and caches the result. "First bounce cache, second bounce thrash!"
//__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float iterations, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
//  ray r;
//  r.origin = eye;
//
//  glm::vec3 A = glm::normalize(glm::cross(view, up));
//  glm::vec3 B = glm::normalize(glm::cross(A, view));
//
//  float tanVert = tan(fov.y*PI/180);
//  float tanHor = tan(fov.x*PI/180);
//
//  float camDistFromScreen = (float)((resolution.y/2.0)/tanVert);
//  glm::vec3 C = view*camDistFromScreen;
//  glm::vec3 M = eye + C;
//
//  //glm::vec3 H = A * (camDistFromScreen * tanHor);
//  //glm::vec3 V = B * (camDistFromScreen * tanVert);
//
//  float sx = (x/(float)(resolution.x-1));
//  float sy = (y/(float)(resolution.y-1));
//  glm::vec3 point = M - A*(resolution.x/2.0f)*(2.0f*sx - 1) - B*(resolution.y/2.0f)*(2.0f*sy - 1);
//  r.direction = glm::normalize(point - eye);
//
//  return r;
  
	////standard camera raycast stuff
 // glm::vec3 E1 = eye;
 // glm::vec3 C = view;
 // glm::vec3 U = up;
 // float fovx = fov.x;
 // float fovy = fov.y;
 // 
 // float CD = glm::length(C);
 // 
 // glm::vec3 A = glm::cross(C, U);
 // glm::vec3 B = glm::cross(A, C);
 // glm::vec3 M = E1+C;
 // glm::vec3 H = (A*float(CD*tan(fovx*(PI/180))))/float(glm::length(A));
 // glm::vec3 V = (B*float(CD*tan(-fovy*(PI/180))))/float(glm::length(B));
 // 
 // float sx = (x)/(resolution.x-1);
 // float sy = (y)/(resolution.y-1);
 // 
 // glm::vec3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
 // glm::vec3 PmE = P-E1;
 // glm::vec3 R = E1 + (float(200)*(PmE))/float(glm::length(PmE));
 // 
 // glm::vec3 direction = glm::normalize(R);
 // //major performance cliff at this point, TODO: find out why!
 // ray r;
 // r.origin = eye;
 // r.direction = direction;
 // return r;
//}

__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float x, float y, glm::vec3 eye, 
	                                            glm::vec3 M, glm::vec3 A, glm::vec3 B, glm::vec3& point)
{
	ray r;
	r.origin = eye;
	//glm::vec3 H = A * (camDistFromScreen * tanHor);
	//glm::vec3 V = B * (camDistFromScreen * tanVert);

	float sx = (x/(float)(resolution.x-1));
	float sy = (y/(float)(resolution.y-1));
	point = M - A*(resolution.x/2.0f)*(2.0f*sx - 1) - B*(resolution.y/2.0f)*(2.0f*sy - 1);
	r.direction = glm::normalize(point - eye);

	return r;
}

__device__ float GetMinimumIntersection(ray& rayToBeTraced, staticGeom* geoms, unsigned int numberOfGeoms, MeshCuda* meshes, int numberOfMeshes,
	                                    unsigned int& minGeomNum, unsigned int& materialId, glm::vec3& minIntersectionPoint, glm::vec3& minNormal)
{
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	float t = -1;

	for (unsigned int geomNum = 0; geomNum < numberOfGeoms; ++geomNum) {
		float tTemp = findIntersection(geoms[geomNum], meshes, numberOfMeshes, rayToBeTraced, intersectionPoint, normal);
		if (tTemp > 0.001f) {
			if (t < 0 || tTemp < t) {
				t = tTemp;
				materialId = geoms[geomNum].materialid;
				minIntersectionPoint = intersectionPoint;
				minNormal = normal;
				minGeomNum = geomNum;
			}
		}
	}

	return t;
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
      color.x = image[index].x;
      color.y = image[index].y;
      color.z = image[index].z;

	  float maxColor = color.x;
	  if (color.y > maxColor)
		maxColor = color.y;
	  if (color.z > maxColor)
		maxColor = color.z;

	  if (maxColor > 1.0f)
		color /= maxColor;

	  color *= 255.0;

		/*
      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      */

      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

// When an object intersection has already been found, this function can be called to update RayPackage
// and call the BSDF function
__device__ void FindIntersectionAndBSDF(RayPackage& rayPackage, material mat, glm::vec3 intersectionPoint, glm::vec3 normal,
	                                    glm::vec3& color, int randomNumber, int iterations)
{
	// Intersection has occured

	// If the ray has intersected a light source, stop recursion and
	// multiply color with light color, else get the color of the material and get the new ray
	// Set the color			
	if (mat.emittance <= 0.001f)
	{
		// Intersection has occured with non emitting object
		thrust::default_random_engine rng(hash(randomNumber));
		thrust::uniform_real_distribution<float> u01(0,1);
				
		ray temp_r;
		temp_r.origin = rayPackage.rayObj.origin;
		temp_r.direction = rayPackage.rayObj.direction;

		glm::vec3 currColor = rayPackage.color;
		int ret = calculateBSDF(temp_r, rayPackage.rayObj, intersectionPoint, normal, currColor, 
						        rayPackage.color, mat, rayPackage.inside, (float)u01(rng), (float)u01(rng), (float)u01(rng));
				
		if (ret == 2)
		{
			// Transmission. This helps in flipping the refractive indices
			rayPackage.inside = !rayPackage.inside;
		}
		rayPackage.rayObj.origin = intersectionPoint;
		rayPackage.isPresent = true;
		// The index will remain the same, so no need to change it
	}
	else
	{
		color = rayPackage.color * mat.color * mat.emittance;
	}
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(RayPackage* rayPackage, glm::vec2 resolution, float iterations, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, unsigned int numberOfGeoms, material* materials, int numberOfMaterials, MeshCuda* meshes, int numberOfMeshes){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	
	unsigned int intersectGeomNum;
	unsigned int materialId;
	glm::vec3 intersectionPoint;
	glm::vec3 normal;

	if (rayPackage[index].isPresent)
	{
		rayPackage[index].isPresent = false;
		float t = GetMinimumIntersection(rayPackage[index].rayObj, geoms, numberOfGeoms, meshes, numberOfMeshes, intersectGeomNum, materialId, intersectionPoint, normal);
		if (t > 0)
		{
			FindIntersectionAndBSDF(rayPackage[index], materials[materialId], intersectionPoint, normal,
				                    colors[rayPackage[index].index], rayPackage[index].index*(iterations-1000)*rayDepth, iterations);
		}
	}
}

// Kernel call for initial ray cast from the camera
__global__ void InitialRayCast(glm::vec2 resolution, float iterations, cameraData cam, int rayDepth, glm::vec3* colors, 
                               staticGeom* geoms, unsigned int numberOfGeoms, material* materials, unsigned int numberOfMaterials, MeshCuda* meshes, int numberOfMeshes,
							   RayPackage* rayPackage, glm::vec3 camM, glm::vec3 camA, glm::vec3 camB, float distImagePlaneFromCamera)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	thrust::default_random_engine rng(hash(index*iterations));
	// Intersection has occured with non emitting object
	thrust::uniform_real_distribution<float> u01(-0.5,0.5);
	thrust::uniform_real_distribution<float> u02(-0.5,0.5);
	float xJittered, yJittered;
	xJittered = (float)x + (float)u01(rng);
	yJittered = (float)y + (float)u02(rng);

	glm::vec3 screenPoint;
	//ray r = raycastFromCameraKernel(resolution, iterations, xJittered, yJittered, cam.position, cam.view, cam.up, cam.fov);
	ray r = raycastFromCameraKernel(resolution, xJittered, yJittered, cam.position, camM, camA, camB, screenPoint);
	
	if (cam.aperture <= 25)
	{
		// For Depth of field
		// Find intersection point of ray with focal plane
		// Generate random points within the aperture and shoot it towards the point found above
		float focalPlaneDist = cam.focalDist;
		glm::vec3 focusPoint;
		glm::vec3 rayDirection = r.direction;
	
		// For intersection with focal plane we can project it in 2D and then find it
		// As the focal plane is parallel to the image plane we can do this in 2 projections and find the intersection point
		// y coordinate of point
	
		focusPoint.x = cam.position.x + rayDirection.x * focalPlaneDist;
		focusPoint.y = cam.position.y + rayDirection.y * focalPlaneDist;
		focusPoint.z = cam.position.z + rayDirection.z * focalPlaneDist;

		//// We know the aperture value (f-Number) from cam.aperture
		////1/f = 1/u + 1/v, u->focalPlane distance from camera, v-> imagePlane distance from camera
		//float u = focusPoint.z-cam.position.z;
		//float v = cam.position.z - screenPoint.z;
		//float focalLength = 1/abs(u) + 1/abs(v);
		//focalLength = 1/focalLength;
		//if (iterations ==1)
		//{
		//	printf("Focal Length: %f\n", focalLength);
		//}

		//float magnification = focalPlaneDist / distImagePlaneFromCamera;
		//float totalDistanceBetweenPlanes = focalPlaneDist + distImagePlaneFromCamera;

		float focalLength = 8.0f;
		// f-Number = focalLength/diameter
		float lensRadius = (focalLength/cam.aperture)/2.0f;
	
		// Generate random numbers in the aperture sphere
		thrust::uniform_real_distribution<float> u03(-lensRadius, lensRadius);
		glm::vec3 rayPointOnCamera = glm::vec3(cam.position.x + (float)u03(rng), cam.position.y + (float)u03(rng), cam.position.z);
		r.origin = rayPointOnCamera;
		r.direction = glm::normalize(focusPoint - rayPointOnCamera);
	
		//// Hyperfocal distance H = f*f/(N*c)
		//float H = f*f/(cam.aperture*0.1);
		//// Depth of focus near point Dnear
		//float Dn = (H*focalPlaneDist)/(H+focalPlaneDist);
		//float Df = (H*focalPlaneDist)/(H-focalPlaneDist);
		//if (iterations == 1 && index <=2)
		//	std::printf("H: %f   Dn: %f Df: %f\n", H, Dn, Df);

	}
	
	unsigned int intersectGeomNum;
	unsigned int materialId;
	glm::vec3 intersectionPoint;
	glm::vec3 normal;

	// Initialize ray Package
	rayPackage[index].rayObj.origin = glm::vec3(0,0,0);
	rayPackage[index].rayObj.direction = glm::vec3(0,0,0);
	rayPackage[index].inside = false;
	rayPackage[index].isPresent = false;
	rayPackage[index].color = glm::vec3(1,1,1);

	// Clear pixel value
	colors[index] = glm::vec3(0,0,0);

	if((x<=resolution.x && y<=resolution.y))
	{
		float t = GetMinimumIntersection(r, geoms, numberOfGeoms, meshes, numberOfMeshes, intersectGeomNum, materialId, intersectionPoint, normal);
		if (t > 0)
		{
			rayPackage[index].rayObj.origin = r.origin;
			rayPackage[index].rayObj.direction = r.direction;

			FindIntersectionAndBSDF(rayPackage[index], materials[materialId], intersectionPoint, normal,
				                    colors[index], index*iterations, iterations);
			
			rayPackage[index].index = index;
		}
	}
}

__global__ void CalculateColor(glm::vec2 resolution, glm::vec3* existingColor, glm::vec3* newColor, int iterations)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if (x<=resolution.x && y<=resolution.y)
	{
		newColor[index] = ((existingColor[index] * (float) (iterations-1)) + newColor[index]) / (float)iterations;
	}
	glm::clamp(newColor[index], 0, 1);
}


__global__ void AssignMeshPointers(MeshCuda* mesh, unsigned int meshNumber, glm::vec3* vertices, unsigned int numVertices, 
	                               Triangle* faces, unsigned int numFaces, glm::vec3* normals, unsigned int numNormals)
{
	//printf("Old Val: %f \n", vertices[0].x);
	(mesh+meshNumber)->vertices = &(vertices[0]);
	mesh[meshNumber].faces = faces;
	mesh[meshNumber].normals = normals;
	mesh[meshNumber].numVertices = numVertices;
	mesh[meshNumber].numFaces = numFaces;
	mesh[meshNumber].numNormals = numNormals;
}

__global__ void AssignMeshPointers1(MeshCuda* mesh, int meshNumber, unsigned int p_numVertices, unsigned int p_numNormals, 
		     unsigned int p_numFaces)
{
	mesh[meshNumber].numVertices = p_numVertices;
	mesh[meshNumber].numFaces = p_numFaces;
	mesh[meshNumber].numNormals = p_numNormals;
}

__global__ void GetMeshPointers(MeshCuda* mesh, unsigned int meshNumber, glm::vec3** vertices, Triangle** faces, glm::vec3** normals)
{
	vertices = &(mesh[meshNumber].vertices);
	faces    = &(mesh[meshNumber].faces);
	normals  = &(mesh[meshNumber].normals);
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, unsigned int iterations, material* materials, unsigned int numberOfMaterials, 
	                  geom* geoms, unsigned int numberOfGeoms, std::vector<Mesh>& meshes){
  
	int traceDepth = 1; //determines how many bounces the raytracer traces
	int totalRays = (int)renderCam->resolution.x*(int)renderCam->resolution.y;

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
	//send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	
	//package geometry and materials and sent to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	for(int i=0; i<numberOfGeoms; ++i){
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.meshIndex = geoms[i].meshIndex;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		
		//// For Motion Blur
		//if (i == 6)
		//{
		//	newStaticGeom.translation.x += 0.0005;
		//	geoms[i].translations[frame] = newStaticGeom.translation;
		//	glm::mat4 transform = utilityCore::buildTransformationMatrix(newStaticGeom.translation, newStaticGeom.rotation, newStaticGeom.scale);
		//	geoms[i].transforms[frame] =  utilityCore::glmMat4ToCudaMat4(transform);
		//	geoms[i].inverseTransforms[frame] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
		//	geoms[i].inverseTransposeTransforms[frame] = utilityCore::glmMat4ToCudaMat4(glm::inverse(glm::transpose(transform)));
		//}

		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		newStaticGeom.inverseTransposeTransform = geoms[i].inverseTransposeTransforms[frame];
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
	cam.aperture = renderCam->aperture;
	cam.focalDist = renderCam->focalDistance;


	MeshCuda* cudaMeshPtr = NULL;
	int numberOfMeshes = meshes.size();
	MeshCuda* temp_cudaMeshPtr;
	if (numberOfMeshes > 0)
	{
		cudaMalloc((void **)&cudaMeshPtr, numberOfMeshes*sizeof(MeshCuda));
		//temp_cudaMeshPtr = new MeshCuda[numberOfMeshes];
		// Get the meshes into the GPU
		for (unsigned int  i=0; i<meshes.size(); ++i)
		{
			unsigned int temp_numVertices = meshes[i].vertices.size();
			glm::vec3* temp_vertices = new glm::vec3[temp_numVertices];
			for (unsigned int currVertex=0; currVertex < temp_numVertices; ++currVertex)
			{
				temp_vertices[currVertex] = (meshes[i]).vertices[currVertex];
			}

			unsigned int temp_numFaces = meshes[i].faces.size();
			Triangle* temp_faces = new Triangle[temp_numFaces];
			for (unsigned int currFace=0; currFace < temp_numFaces; ++currFace)
			{
				temp_faces[currFace] = (meshes[i]).faces[currFace];
			}

			unsigned int temp_numNormals = meshes[i].normals.size();
			glm::vec3* temp_normals = new glm::vec3[temp_numNormals];
			for (unsigned int currNormal=0; currNormal < temp_numNormals; ++currNormal)
			{
				temp_normals[currNormal] = (meshes[i]).normals[currNormal];
			}


			
			temp_cudaMeshPtr = new MeshCuda(temp_vertices, temp_numVertices, temp_normals, temp_numNormals,
				                           temp_faces, temp_numFaces);;

			cudaMemcpy(cudaMeshPtr+i, temp_cudaMeshPtr, sizeof(MeshCuda), cudaMemcpyHostToDevice);
		}
	}

	RayPackage* cudaRayPackage = NULL;
	cudaMalloc((void**)&cudaRayPackage, totalRays*sizeof(RayPackage));

	glm::vec3 M, A, B;
	float distImagePlaneFromCamera;
	GetParametersForRayCast(renderCam->resolution, renderCam->positions[frame], renderCam->views[frame], renderCam->ups[frame], 
		                    renderCam->fov, M, A, B, distImagePlaneFromCamera);
	
	//kernel launches
	InitialRayCast<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, 
																				numberOfMaterials, cudaMeshPtr, numberOfMeshes, cudaRayPackage, M, A, B, distImagePlaneFromCamera);
	int rayCount = totalRays;
	dim3 fullBlocksPerGridNew;
	++traceDepth;
	while (traceDepth <= MAX_DEPTH)
	{
		// wrap raw pointer with a device_ptr for Stream compaction
		thrust::device_ptr<RayPackage> devRayPackagePtr(cudaRayPackage);
		thrust::device_ptr<RayPackage> devRayPackageEndPtr = thrust::remove_if(devRayPackagePtr, devRayPackagePtr+rayCount, IsRayAbsent());
		rayCount = devRayPackageEndPtr.get() - devRayPackagePtr.get();

		if (rayCount <= 0)
		    break;
		
		// Create blocks for lesser number of rays found by stream compaction
		fullBlocksPerGridNew = dim3((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(rayCount)/(float)renderCam->resolution.x/float(tileSize)));
		raytraceRay<<<fullBlocksPerGridNew, threadsPerBlock>>>(cudaRayPackage, renderCam->resolution, (float)iterations, traceDepth, cudaimage, cudageoms, numberOfGeoms, 
			                                                   cudamaterials, numberOfMaterials, cudaMeshPtr, numberOfMeshes);
		
		++traceDepth;
	}

	glm::vec3* cudaExistingColor = NULL;
	cudaMalloc((void**)&cudaExistingColor, totalRays*sizeof(glm::vec3));
	cudaMemcpy( cudaExistingColor, renderCam->image, totalRays*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	CalculateColor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaExistingColor, cudaimage, iterations);
	cudaFree( cudaExistingColor );
	
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, totalRays*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	//free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudamaterials );
	cudaFree( cudaRayPackage );
	delete [] geomList;

	// Clear Mesh Data
	for (unsigned int  i=0; i<meshes.size(); ++i)
	{
		glm::vec3* cudaMeshVerticesPtr = NULL;
		Triangle* cudaMeshFacesPtr = NULL;
		glm::vec3* cudaMeshNormalsPtr = NULL;
		GetMeshPointers<<<1,1>>>(cudaMeshPtr, i, &cudaMeshVerticesPtr, &cudaMeshFacesPtr, &cudaMeshNormalsPtr);

		cudaFree(cudaMeshVerticesPtr);
		cudaFree(cudaMeshFacesPtr);
		cudaFree(cudaMeshNormalsPtr);
	}

	if (meshes.size() > 0)
	{
		cudaFree(cudaMeshPtr);
	}

	// make certain the kernel has completed 
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
