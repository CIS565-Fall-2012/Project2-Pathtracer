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
__host__ __device__ float boxIntersectionTest(glm::vec3 boxMin, glm::vec3 boxMax, staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
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
    if(fabs(fabs(a)-fabs(b))<0.000000001){
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

//Wrapper for cube intersection test for testing against unit cubes
__host__ __device__  float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  return boxIntersectionTest(glm::vec3(-.5,-.5,-.5), glm::vec3(.5,.5,.5), box, r, intersectionPoint, normal);
}

//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float boxIntersectionTest(glm::vec3 boxMin, glm::vec3 boxMax, staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
    //glm::vec3 currentNormal = glm::vec3(0,0,0);

    //ray ro = r;

    //glm::vec3 iP0 = multiplyMV(box.inverseTransform,glm::vec4(r.origin, 1.0f));
    //glm::vec3 iP1 = multiplyMV(box.inverseTransform,glm::vec4(r.origin+r.direction, 1.0f));
    //glm::vec3 iV0 = iP1 - iP0;

    //r.origin = iP0; 
    //r.direction = glm::normalize(iV0);

    //float tmin, tmax, tymin, tymax, tzmin, tzmax;

    //glm::vec3 rsign = getSignOfRay(r);
    //glm::vec3 rInverseDirection = getInverseDirectionOfRay(r);

    //if((int)rsign.x==0){
    //  tmin = (boxMin.x - r.origin.x) * rInverseDirection.x;
    //  tmax = (boxMax.x - r.origin.x) * rInverseDirection.x;
    //}else{
    //  tmin = (boxMax.x - r.origin.x) * rInverseDirection.x;
    //  tmax = (boxMin.x - r.origin.x) * rInverseDirection.x;
    //}

    //if((int)rsign.y==0){
    //  tymin = (boxMin.y - r.origin.y) * rInverseDirection.y;
    //  tymax = (boxMax.y - r.origin.y) * rInverseDirection.y;
    //}else{
    //  tymin = (boxMax.y - r.origin.y) * rInverseDirection.y;
    //  tymax = (boxMin.y - r.origin.y) * rInverseDirection.y;
    //}

    //if ( (tmin > tymax) || (tymin > tmax) ){
    //    return -1;
    //}
    //if (tymin > tmin){
    //    tmin = tymin;
    //}
    //if (tymax < tmax){
    //    tmax = tymax;
    //}

    //if((int)rsign.z==0){
    //  tzmin = (boxMin.z - r.origin.z) * rInverseDirection.z;
    //  tzmax = (boxMax.z - r.origin.z) * rInverseDirection.z;
    //}else{
    //  tzmin = (boxMax.z - r.origin.z) * rInverseDirection.z;
    //  tzmax = (boxMin.z - r.origin.z) * rInverseDirection.z;
    //}

    //if ( (tmin > tzmax) || (tzmin > tmax) ){
    //    return -1;
    //}
    //if (tzmin > tmin){
    //    tmin = tzmin;
    //}
    //if (tzmax < tmax){
    //    tmax = tzmax;
    //}
    //if(tmin<0){
    //    return -1;
    //}

    //glm::vec3 osintersect = r.origin + tmin*r.direction;

    //if(abs(osintersect.x-abs(boxMax.x))<.001){
    //    currentNormal = glm::vec3(1,0,0);
    //}else if(abs(osintersect.y-abs(boxMax.y))<.001){
    //    currentNormal = glm::vec3(0,1,0);
    //}else if(abs(osintersect.z-abs(boxMax.z))<.001){
    //    currentNormal = glm::vec3(0,0,1);
    //}else if(abs(osintersect.x+abs(boxMin.x))<.001){
    //    currentNormal = glm::vec3(-1,0,0);
    //}else if(abs(osintersect.y+abs(boxMin.y))<.001){
    //    currentNormal = glm::vec3(0,-1,0);
    //}else if(abs(osintersect.z+abs(boxMin.z))<.001){
    //    currentNormal = glm::vec3(0,0,-1);
    //}

    //intersectionPoint = multiplyMV(box.transform, glm::vec4(osintersect, 1.0));



    //normal = multiplyMV(box.transform, glm::vec4(currentNormal,0.0));
    //return glm::length(intersectionPoint-ro.origin);

		glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));

	ray rt; rt.origin = ro; rt.direction = rd;

	glm::vec3 bl, bh;
	bl = glm::vec3(-0.5, -0.5, -0.5);
	bh = glm::vec3(0.5, 0.5, 0.5);

	double tnear, tfar, t1, t2, temp;
	tnear = -99999.0;
	tfar = 99999.0;

	if(abs(rt.direction.x) <= 0.001 && (rt.origin.x < bl.x || rt.origin.x > bh.x))
		return -1;

	else
	{
		t1 = (bl.x - rt.origin.x)/rt.direction.x;
		t2 = (bh.x - rt.origin.x)/rt.direction.x;

		if(t1 > t2)
		{
			temp = t1;
			t1 = t2;
			t2 = temp;
		}

		if(t1 > tnear)
			tnear = t1;

		if(t2 < tfar)
			tfar = t2;

		if(tnear > tfar)
			return -1;

		if(tfar < 0.001)
			return -1;
	}

	if(abs(rt.direction.y) <= 0.001 && (rt.origin.y < bl.y || rt.origin.y > bh.y))
		return -1;

	else
	{
		t1 = (bl.y - rt.origin.y)/rt.direction.y;
		t2 = (bh.y - rt.origin.y)/rt.direction.y;

		if(t1 > t2)
		{
			temp = t1;
			t1 = t2;
			t2 = temp;
		}

		if(t1 > tnear)
			tnear = t1;

		if(t2 < tfar)
			tfar = t2;

		if(tnear > tfar)
			return -1;

		if(tfar < 0.001)
			return -1;
	}

	if(abs(rt.direction.z) <= 0.001 && (rt.origin.z < bl.z || rt.origin.z > bh.z))
		return -1;

	else
	{
		t1 = (bl.z - rt.origin.z)/rt.direction.z;
		t2 = (bh.z - rt.origin.z)/rt.direction.z;

		if(t1 > t2)
		{
			temp = t1;
			t1 = t2;
			t2 = temp;
		}

		if(t1 > tnear)
			tnear = t1;

		if(t2 < tfar)
			tfar = t2;

		if(tnear > tfar)
			return -1;

		if(tfar < 0.001)
			return -1;
	}

	if(abs(tnear) < 0.001)
		return -1;

	if(tnear < tfar)
	{
		glm::vec3 point = glm::vec3(rt.origin.x + tnear*rt.direction.x, rt.origin.y + tnear*rt.direction.y, rt.origin.z + tnear*rt.direction.z);
		
		if(fabs( point.x - 0.5 ) < 0.001)
			normal = glm::vec3(1.0, 0.0, 0.0);
		
		if(fabs( point.x + 0.5 ) < 0.001)
			normal = glm::vec3(-1.0, 0.0, 0.0);
		
		if(fabs( point.y - 0.5 ) < 0.001)
			normal = glm::vec3(0.0, 1.0, 0.0);
		
		if(fabs( point.y + 0.5 ) < 0.001)
			normal = glm::vec3(0.0, -1.0, 0.0);
		
		if(fabs( point.z - 0.5 ) < 0.001)
			normal = glm::vec3(0.0, 0.0, 1.0);
		
		if(fabs( point.z + 0.5 ) < 0.001)
			normal = glm::vec3(0.0, 0.0, -1.0);

		normal =  glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0)));
		//normal = glm::normalize(normal);
		glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, tnear), 1.0));
		intersectionPoint = realIntersectionPoint;
		return glm::length(r.origin - realIntersectionPoint);
	}
	
	if(abs(tfar) < 0.001)
		return -1;

	if(tfar > 0.001)
	{
		glm::vec3 point = glm::vec3(rt.origin.x + tfar*rt.direction.x, rt.origin.y + tfar*rt.direction.y, rt.origin.z + tfar*rt.direction.z);
		
		if(fabs( point.x - 0.5 ) < 0.001)
			normal = glm::vec3(1.0, 0.0, 0.0);
		
		if(fabs( point.x + 0.5 ) < 0.001)
			normal = glm::vec3(-1.0, 0.0, 0.0);
		
		if(fabs( point.y - 0.5 ) < 0.001)
			normal = glm::vec3(0.0, 1.0, 0.0);
		
		if(fabs( point.y + 0.5 ) < 0.001)
			normal = glm::vec3(0.0, -1.0, 0.0);
		
		if(fabs( point.z - 0.5 ) < 0.001)
			normal = glm::vec3(0.0, 0.0, 1.0);
		
		if(fabs( point.z + 0.5 ) < 0.001)
			normal = glm::vec3(0.0, 0.0, -1.0);
		
		normal =  glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0)));
		//normal = glm::normalize(normal);
		glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, tfar), 1.0));
		intersectionPoint = realIntersectionPoint;
		return glm::length(r.origin - realIntersectionPoint);
	}

    return -1;
}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
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

//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){

  float radius=.5f;
  thrust::default_random_engine rng(hash(randomSeed));
  thrust::uniform_real_distribution<float> u01(0,1);
  thrust::uniform_real_distribution<float> u02(-0.5,0.5);

  glm::vec3 point = glm::vec3(0,0,0);
  float x=(float)u02(rng);
  float y=(float)u02(rng);
  float z=0;
  float russianRoulette = (float)u01(rng); 
  if(russianRoulette<0.5){
    z=(float)sqrt(radius*radius-x*x-y*y);
  }else
    z=-(float)sqrt(radius*radius-x*x-y*y);

  point=glm::vec3(x,y,z);
  glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(point,1.0f));

  return randPoint;
}

#endif