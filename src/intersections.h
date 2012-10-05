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
__host__ __device__ float boxIntersectionTest(const staticGeom &box, const ray &r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(const staticGeom &sphere, const ray &r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

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
__host__ __device__  float boxIntersectionTest(const staticGeom &box, const ray &r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
	ray rt;
	rt.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
	rt.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));

	glm::vec3 bounds[2];
	bounds[0] = glm::vec3(-0.5, -0.5, -0.5);
	bounds[1] = glm::vec3(0.5, 0.5, 0.5);

	glm::vec3 invDirection(1.0f / rt.direction.x, 1.0f / rt.direction.y, 1.0f / rt.direction.z);
	int sign[3];
	sign[0] = (invDirection.x < 0.0f);
	sign[1] = (invDirection.y < 0.0f);
	sign[2] = (invDirection.z < 0.0f);
	unsigned int axisMin;	//0 = X, 1 = Y, 2 = Z
	unsigned int axisMax;

	float tmin = -1e6, tmax = 1e6, txmin, txmax, tymin, tymax, tzmin, tzmax;
	txmin = (bounds[sign[0]].x - rt.origin.x) * invDirection.x;
	txmax = (bounds[1 - sign[0]].x - rt.origin.x) * invDirection.x;

	if(txmin < 0 && txmax < 0)
	{
		return -1.0f;
	}

	if(txmin > tmin && txmin > 0)
	{
		tmin = txmin;
		axisMin = 0;
	}

	if(txmax < tmax && txmax > 0)
	{
		tmax = txmax;
		axisMax = 0;
	}

	tymin = (bounds[sign[1]].y - rt.origin.y) * invDirection.y;
	tymax = (bounds[1 - sign[1]].y - rt.origin.y) * invDirection.y;

	if(tymin < 0 && tymax < 0)
	{
		return -1.0f;
	}

	if((tmin > tymax) || (tymin > tmax))
		return -1.0f;

	if(tymin > tmin && tymin > 0)
	{
		tmin = tymin;
		axisMin = 1;
	}

	if(tymax < tmax && tymax > 0)
	{
		tmax = tymax;
		axisMax = 1;
	}

	tzmin = (bounds[sign[2]].z - rt.origin.z) * invDirection.z;
	tzmax = (bounds[1 - sign[2]].z - rt.origin.z) * invDirection.z;

	if(tzmin < 0 && tzmax < 0)
	{
		return -1.0f;
	}

	if((tmin > tzmax) || (tzmin > tmax))
		return -1.0f;

	if(tzmin > tmin && tzmin > 0)
	{
		tmin = tzmin;
		axisMin = 2;
	}

	if(tzmax < tmax && tzmax > 0)
	{
		tmax = tzmax;
		axisMax = 2;
	}

	/*
	intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, tmin), 1.0));
	if(axisMin == 0)
	{
		normal = glm::vec3(-1, 0, 0);
		if(sign[0])
		{
			normal.x = 1.0f;
		}
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
	}
	else if(axisMin == 1)
	{
		normal = glm::vec3(0, -1, 0);
		if(sign[1])
		{
			normal.y = 1.0f;
		}
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
	}
	else
	{
		normal = glm::vec3(0, 0, -1);
		if(sign[2])
		{
			normal.z = 1.0f;
		}
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
	}
	*/

	if(!epsilonCheck(tmin, -1e6f))
	{
		intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, tmin), 1.0));
		if(axisMin == 0)
		{
			normal = glm::vec3(-1, 0, 0);
			if(sign[0])
			{
				normal.x = 1.0f;
			}
			normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
		}
		else if(axisMin == 1)
		{
			normal = glm::vec3(0, -1, 0);
			if(sign[1])
			{
				normal.y = 1.0f;
			}
			normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
		}
		else
		{
			normal = glm::vec3(0, 0, -1);
			if(sign[2])
			{
				normal.z = 1.0f;
			}
			normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
		}
	}
	else if(!epsilonCheck(tmax, 1e6f))
	{
		intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, tmax), 1.0));
		if(axisMax == 0)
		{
			normal = glm::vec3(-1, 0, 0);
			if(sign[0])
			{
				normal.x = 1.0f;
			}
			normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
		}
		else if(axisMax == 1)
		{
			normal = glm::vec3(0, -1, 0);
			if(sign[1])
			{
				normal.y = 1.0f;
			}
			normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
		}
		else
		{
			normal = glm::vec3(0, 0, -1);
			if(sign[2])
			{
				normal.z = 1.0f;
			}
			normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
		}
	}
	else
	{
		return -1.0f;
	}

	return glm::distance(r.origin, intersectionPoint);
}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float sphereIntersectionTest(const staticGeom &sphere, const ray &r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;

  ray rt;
  rt.origin = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  rt.direction = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));
  /*
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  */

  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0){
    return -1;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
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
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed)
{
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(-1.0f, 1.0f);
	float x1, x2;
	do
	{
		x1 = (float)u01(rng);
		x2 = (float)u01(rng);
	}while(x1 * x1 + x2 * x2 >= 1.0f);

	return glm::vec3(2.0f * x1 * sqrt(1.0f - x1 * x1 - x2 * x2), 2.0f * x2 * sqrt(1.0f - x1 * x1 - x2 * x2), 1.0f - 2 * sqrt(x1 * x1 + x2 * x2));
}

#endif