// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);
__host__ __device__ float TriangleArea(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3);
__host__ __device__  float MeshIntersectionTest(staticGeom meshObj, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
//__host__ __device__ float Test_RayPolyIntersect(glm::vec3 const& P0, glm::vec3 const& V0, glm::vec3 const& p1, glm::vec3 const& p2, glm::vec3 const& p3, cudaMat4 const& T, cudaMat4 const& inverseT);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	return a;
}

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b){
	if(fabs(fabs(a)-fabs(b))<EPSILON){
		return true;
	}else{
		return false;
	}
}

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
  return r.origin + float(t-.0001)*glm::normalize(r.direction);
}

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors. 
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
		
  glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  glm::vec3 vertex1(-0.5f, -0.5f, -0.5f);
  glm::vec3 vertex2(+0.5f, +0.5f, +0.5f);
  bool inside = false;
  //Check if point is inside the cube
	if(rt.origin.x > vertex1.x && rt.origin.x < vertex2.x && rt.origin.y > vertex1.y && rt.origin.y < vertex2.y && rt.origin.z > vertex1.z && rt.origin.z < vertex2.z)
	{
		r.origin = r.origin - r.direction * 100000.0f;
		inside = true;
	}

  float t, t1, t2;
  float temp;
  float tFar = 1000000.0f;
  float tNear = -10000000.0f;

  if(rd.x == 0)
  {
	  if(!(ro.x >= vertex1.x && ro.x <= vertex2.x))
		  return -1;
  }
  else
  {
	  t1 = (vertex1.x - ro.x) / rd.x;
	  t2 = (vertex2.x - ro.x) / rd.x;

	  if(t1 > t2)
	  {
		  temp = t1;
		  t1 = t2;
		  t2 = temp;
	  }
	  if(t1 > tNear)
		  tNear = t1;
	  if(t2 < tFar)
		  tFar = t2;
	  if(tNear > tFar)
		  return -1;
	  if(tFar < 0)
		  return -1;
  }

  if(rd.y == 0)
  {
	   if(!(ro.y >= vertex1.y && ro.y <= vertex2.y))
		  return -1;
  }
  else
  {
	  t1 = (vertex1.y - ro.y) / rd.y;
	  t2 = (vertex2.y - ro.y) / rd.y;

	  if(t1 > t2)
	  {
		  temp = t1;
		  t1 = t2;
		  t2 = temp;
	  }
	  if(t1 > tNear)
		  tNear = t1;
	  if(t2 < tFar)
		  tFar = t2;
	  if(tNear > tFar)
		  return -1;
	  if(tFar < 0)
		  return -1;
  }

  if(rd.z == 0)
  {
	   if(!(ro.z >= vertex1.z && ro.z <= vertex2.z))
		  return -1;
  }
  else
  {
	  t1 = (vertex1.z - ro.z) / rd.z;
	  t2 = (vertex2.z - ro.z) / rd.z;

	  if(t1 > t2)
	  {
		  temp = t1;
		  t1 = t2;
		  t2 = temp;
	  }
	  if(t1 > tNear)
		  tNear = t1;
	  if(t2 < tFar)
		  tFar = t2;
	  if(tNear > tFar)
		  return -1;
	  if(tFar < 0)
		  return -1;
  }

  if(tNear>= -100000.0f)
	  t = tNear;
  else
	  return -1;

	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, t), 1.0f));
	glm::vec3 realOrigin = multiplyMV(box.transform, glm::vec4(0,0,0,1));

	intersectionPoint = realIntersectionPoint;

	glm::vec3 P(0.0,0.0,0.0);
	normal = glm::vec3(0.0,0.0,0.0);
	P = multiplyMV(box.inverseTransform, glm::vec4(realIntersectionPoint, 1.0f));


	if(P.x >= 0.5 - 0.005 && P.x <= 0.5 + 0.005)
		normal = glm::vec3(1.0f, 0.0f, 0.0f);
	else if(P.x >= -0.5 - 0.005 && P.x <= -0.5 + 0.005)
		normal = glm::vec3(-1.0f, 0.0f, 0.0f);
	else if(P.y >= 0.5 - 0.005 && P.y <= 0.5 + 0.005)
		normal = glm::vec3(0.0f, 1.0f, 0.0f);
	else if(P.y >= -0.5 - 0.005 && P.y <= -0.5 + 0.005)
		normal = glm::vec3(0.0f, -1.0f, 0.0f);
	else if(P.z >= 0.5 - 0.005 && P.z <= 0.5 + 0.005)
		normal = glm::vec3(0.0f, 0.0f, 1.0f);
	else if(P.z >= -0.5 - 0.005 && P.z <= -0.5 + 0.005)
		normal = glm::vec3(0.0f, 0.0f, -1.0f);

	normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
	
	//The way I handle the case of point inside the cube gives the correct internal normal
	//if(inside)
		//normal = -normal;

	return glm::length(r.origin - realIntersectionPoint);
}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
//Returns intersectionPoint and normal - as return by reference and Distance to Instersection as return float
__host__ __device__  float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
	float radius = .5;
		
	glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

	ray rt; rt.origin = ro; rt.direction = rd;
	bool inside = false;
	//Check if point is inside sphere
	if(glm::length(rt.origin - glm::vec3(0.0, 0.0, 0.0)) < radius)
		inside = true;

	//Ax2 + Bx + C = 0
	float vDotDirection = glm::dot(rt.origin, rt.direction);   //Gives B
	float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2)); //B^2 - 4*A*C
	if (radicand < 0){
	return -1;
	}
  
	float squareRoot = sqrt(radicand);
	float firstTerm = -vDotDirection; //-B
	float t1 = firstTerm + squareRoot; //-B + radicand
	float t2 = firstTerm - squareRoot; //-B - radicand
  
	float t = 0;
	if (t1 < 0 && t2 < 0) {
		return -1;
	} else if (t1 > 0 && t2 > 0) {
		t = min(t1, t2);
	} else {
		t = max(t1, t2);
	}     

	glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
	glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

	intersectionPoint = realIntersectionPoint;
	normal = glm::normalize(realIntersectionPoint - realOrigin);   
	
	//If point is inside, return the inward normal, which is the negative of the outward normal
	if(inside)
		normal = glm::normalize(-normal);

	return glm::length(r.origin - realIntersectionPoint);
}

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
	glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
	glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
	glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
	glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
	float xradius = glm::distance(origin, xmax);
	float yradius = glm::distance(origin, ymax);
	float zradius = glm::distance(origin, zmax);
	return glm::vec3(xradius, yradius, zradius);
}

//LOOK: Example for generating a random point on an object using thrust. 
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);
	thrust::uniform_real_distribution<float> u02(-0.5,0.5);

	//get surface areas of sides
	glm::vec3 radii = getRadiuses(cube);
	float side1 = radii.x * radii.y * 4.0f;   //x-y face
	float side2 = radii.z * radii.y * 4.0f;   //y-z face
	float side3 = radii.x * radii.z* 4.0f;   //x-z face
	float totalarea = 2.0f * (side1+side2+side3);
	
	//pick random face, weighted by surface area
	float russianRoulette = (float)u01(rng);
	
	glm::vec3 point = glm::vec3(.5,.5,.5);
	
	if(russianRoulette<(side1/totalarea)){
		//x-y face
		point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
	}else if(russianRoulette<((side1*2)/totalarea)){
		//x-y-back face
		point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
	}else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
		//y-z face
		point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
	}else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
		//y-z-back face
		point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
	}else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
		//x-z face
		point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
	}else{
		//x-z-back face
		point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
	}
	
	glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

	return randPoint;
	   
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> U(0,1);
	thrust::uniform_real_distribution<float> V(0,1);
	float Theta = TWO_PI * U(rng);
	float Phi = acos(2*V(rng) -1);
	
	float x = cos(Theta) * sin(Phi);
	float y = sin(Theta) * sin(Phi);
	float z = cos(Phi);
	
	glm::vec3 PointOnSphere = sphere.translation + glm::vec3(x,y,z);
	return PointOnSphere;
}

__host__ __device__  float MeshIntersectionTest(staticGeom meshObj, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
	float selectedt = 10000000.0;
	int triangleIndex = -1;
	ray rt;
	bool intersected = false;
	rt.origin = multiplyMV(meshObj.inverseTransform, glm::vec4(r.origin, 1.0));
	rt.direction = glm::normalize(multiplyMV(meshObj.inverseTransform, glm::vec4(r.direction, 0.0)));
/*	
	glm::vec3 maxBox = glm::vec3(abs(meshObj.triangles[0].vertices[0].x), abs(meshObj.triangles[0].vertices[0].y), abs(meshObj.triangles[0].vertices[0].z));
	glm::vec3 minBox = glm::vec3(-abs(meshObj.triangles[0].vertices[0].x), -abs(meshObj.triangles[0].vertices[0].y), -abs(meshObj.triangles[0].vertices[0].z));
	//Axis Aligned Bounding Sphere (Not scaled equally on all axis)
	for(int i = 0; i < meshObj.numOfTriangles; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			if(meshObj.triangles[i].vertices[j].x > maxBox.x)
				maxBox.x = meshObj.triangles[i].vertices[j].x + 0.1;

			if(meshObj.triangles[i].vertices[j].y > maxBox.y)
				maxBox.y = meshObj.triangles[i].vertices[j].y + 0.1;

			if(meshObj.triangles[i].vertices[j].z > maxBox.z)
				maxBox.z = meshObj.triangles[i].vertices[j].z + 0.1;

			if(meshObj.triangles[i].vertices[j].x < minBox.x)
				minBox.x = meshObj.triangles[i].vertices[j].x - 0.1;

			if(meshObj.triangles[i].vertices[j].y < minBox.y)
				minBox.y = meshObj.triangles[i].vertices[j].y - 0.1;

			if(meshObj.triangles[i].vertices[j].z < minBox.z)
				minBox.z = meshObj.triangles[i].vertices[j].z - 0.1;
		}
	}
	staticGeom BoundingBall;
	BoundingBall.type = SPHERE;
	BoundingBall.rotation = glm::vec3(0.0, 0.0, 0.0);
	BoundingBall.translation = (maxBox + minBox) * 0.5f;
	if(abs(maxBox.x - BoundingBall.translation.x) > abs(minBox.x - BoundingBall.translation.x))
		BoundingBall.scale.x = 2.0 * abs(maxBox.x - BoundingBall.translation.x);
	else
		BoundingBall.scale.x = 2.0 * abs(minBox.x - BoundingBall.translation.x);

	if(abs(maxBox.y - BoundingBall.translation.y) > abs(minBox.y - BoundingBall.translation.y))
		BoundingBall.scale.y = 2.0 * abs(maxBox.y - BoundingBall.translation.y);
	else
		BoundingBall.scale.y = 2.0 * abs(minBox.y - BoundingBall.translation.y);

	if(abs(maxBox.z - BoundingBall.translation.z) > abs(minBox.z - BoundingBall.translation.z))
		BoundingBall.scale.z = 2.0 * abs(maxBox.z - BoundingBall.translation.z);
	else
		BoundingBall.scale.z = 2.0 * abs(minBox.z - BoundingBall.translation.z);

	glm::mat4 transform = utilityCore::buildTransformationMatrix(BoundingBall.translation, BoundingBall.rotation, BoundingBall.scale);
	transform = utilityCore::cudaMat4ToGlmMat4(meshObj.transform) * transform;
	BoundingBall.transform = utilityCore::glmMat4ToCudaMat4(transform);
	BoundingBall.inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	*/
	staticGeom BoundingBall;
	//BoundingBall.type = SPHERE;
	BoundingBall.type = CUBE;
	BoundingBall.rotation = meshObj.BSrotation;
	BoundingBall.translation = meshObj.BStranslation;
	BoundingBall.scale = meshObj.BSscale;
	BoundingBall.transform = meshObj.BSTransform;
	BoundingBall.inverseTransform = meshObj.BSInverseTransform;
	glm::vec3 tempIP(0,0,0);
	glm::vec3 tempNormal(0.0, 0.0, 0.0);
	float tempT = boxIntersectionTest(BoundingBall, r, tempIP, tempNormal);
	if(tempT == -1)
	{
		//printf("Bounding Ball not intersected\n");
		return -1;
	}
	else
	{
		//printf("Bounding Ball intersected\n");
	}
	
	for(int i = 0; i < meshObj.numOfTriangles; i++)
	{
		float t = -1;
		glm::vec3 p1 = meshObj.triangles[i].vertices[0];
		glm::vec3 p2 = meshObj.triangles[i].vertices[1];
		glm::vec3 p3 = meshObj.triangles[i].vertices[2];
		
		float A, B, C, D;
		float s, s1, s2, s3;
		
		A = (p1.y * (p2.z - p3.z)) + (p2.y * (p3.z - p1.z)) + (p3.y * (p1.z - p2.z));
		B = (p1.z * (p2.x - p3.x)) + (p2.z * (p3.x - p1.x)) + (p3.z * (p1.x - p2.x));
		C = (p1.x * (p2.y - p3.y)) + (p2.x * (p3.y - p1.y)) + (p3.x * (p1.y - p2.y));
		D = -((p1.x * (p2.y * p3.z - p3.y * p2.z)) + (p2.x * (p3.y * p1.z - p1.y * p3.z)) + (p3.x * (p1.y * p2.z - p2.y * p1.z)));
		// AX + BY + CZ + D = 0
		
		glm::vec4 plane = glm::vec4(A, B, C, D);
		glm::vec3 N = glm::vec3(A, B, C);
		
		t = -(D + A*rt.origin.x + B*rt.origin.y + C*rt.origin.z) / (A*rt.direction.x + B*rt.direction.y + C*rt.direction.z); 
		
		if(t < 0)
			continue;
		else
		{
			glm::vec4 Pt = glm::vec4(glm::vec3(rt.origin) + t * glm::vec3(rt.direction), 1.0);
	
			s = TriangleArea(p1, p2, p3);
			s1 = TriangleArea(glm::vec3(Pt), p2, p3)/s;
			s2 = TriangleArea(glm::vec3(Pt), p3, p1)/s;
			s3 = TriangleArea(glm::vec3(Pt), p1, p2)/s;
			
			if( 0 <= s1 && s1 <= 1.0 && 0 <= s2 && s2 <= 1.0 && 0 <= s3 && s3 <= 1.0 && (s1 + s2 + s3 >= 1.0 - 0.0001) && (s1 + s2 + s3 <= 1.0 + 0.0001))
			{
				//
			}
			else 
				continue;
		}
		
		if(t > 0 && t < selectedt)
		{
			selectedt = t - 0.001f;
			triangleIndex = i;
			intersected = true;
			//printf("Intersected\n");
		}
		
	}
	if(triangleIndex < 0)
		return -1;

	glm::vec3 realIntersectionPoint = glm::vec3(multiplyMV(meshObj.transform, glm::vec4(getPointOnRay(rt, selectedt), 1.0)));

	intersectionPoint = realIntersectionPoint;

	//Interpolating Normals
	/*glm::vec3 PointOnTriangle =  getPointOnRay(rt, selectedt);
	float d1 = glm::length(PointOnTriangle - meshObj.triangles[triangleIndex].vertices[0]);
	float d2 = glm::length(PointOnTriangle - meshObj.triangles[triangleIndex].vertices[1]);
	float d3 = glm::length(PointOnTriangle - meshObj.triangles[triangleIndex].vertices[2]);

	float D1 = d1 / (d1 + d2 + d3);
	float D2 = d2 / (d1 + d2 + d3);
	float D3 = d3 / (d1 + d2 + d3);
	
	float D1 = 2.0/3.0f;
	float D2 = 2.0/3.0f;
	float D3 = 2.0/3.0f;
	glm::vec3 NormalAtP = (1-D1) * meshObj.triangles[triangleIndex].normal[0]
							+ (1-D2) * meshObj.triangles[triangleIndex].normal[1]
							+ (1-D3) * meshObj.triangles[triangleIndex].normal[2];
							*/
	//normal = glm::normalize(multiplyMV(meshObj.transform, glm::vec4(glm::normalize(NormalAtP), 0.0)));   
	
	normal = glm::normalize(multiplyMV(meshObj.transform, glm::vec4(glm::normalize(meshObj.triangles[triangleIndex].normal[2]), 0.0)));   
	
	if(glm::dot(normal, -r.direction) < 0)
		normal = glm::normalize(-normal);

	return glm::length(r.origin - realIntersectionPoint);
}

__host__ __device__ float TriangleArea(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
{
	float t1 = 0, t2 = 0 , t3 = 0, s = 0;
	t1 = (p1.y * (p2.z - p3.z)) - (p1.z * (p2.y - p3.y)) + (1 * ((p2.y * p3.z) - (p3.y * p2.z)));
	t2 = (p1.z * (p2.x - p3.x)) - (p1.x * (p2.z - p3.z)) + (1 * ((p2.z * p3.x) - (p3.z * p2.x)));
	t3 = (p1.x * (p2.y - p3.y)) - (p1.y * (p2.x - p3.x)) + (1 * ((p2.x * p3.y) - (p3.x * p2.y)));
	s = 0.5 * sqrt(t1*t1 + t2*t2 + t3*t3);
	return s;
}

#endif