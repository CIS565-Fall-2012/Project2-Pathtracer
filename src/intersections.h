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

using namespace glm;

//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 getPointOnRayUnnormalized(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float boxIntersectionTest(glm::vec3 boxMin, glm::vec3 boxMax, staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash1(unsigned int a){
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
	return r.origin + float(t)*(r.direction);
}


__host__ __device__ glm::vec3 getPointOnRayUnnormalized(ray r, float t){
  return r.origin + float(t)*(r.direction);
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


__host__ __device__ double Test_RayPlaneIntersect(glm::vec3 origin, glm::vec3 direction, glm::vec3 const& p, glm::vec3 const& normal)
{
	// If the ray is parallel to the  plane and there are either no solutions or an infinite number (line in plane).
	// Returning -1 in that case
	double dotProd = glm::dot(normal, direction);
	if (dotProd == 0.0)
		return -1;

	double t = glm::dot(normal, (p-origin)) / dotProd;

	// No intersectionin the viewing direction if t is -ve
	if (abs(t) < 1e-3) // on the surface
		return 0;
	else if (t < 0.0)
		return -1;

	return t;
}


__host__ __device__ double RayTriangleIntersect(staticGeom mesh, ray const& r, glm::vec3 const& p1, glm::vec3 const& p2, glm::vec3 const& p3,
	glm::vec3 normal, glm::vec3& intersectionPoint, glm::vec3& surfaceNormal)
{
	//glm::vec4 P0hom = glm::inverse(T) * glm::vec4(P0, 1.0f);
	//glm::vec4 V0hom = glm::inverse(T) * glm::vec4(V0, 0.0f);
	
	glm::vec3 P0hom = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 V0hom = multiplyMV(mesh.inverseTransform, glm::vec4(glm::normalize(r.direction), 0.0f));

	//glm::vec4 Rd = glm::vec4(V0hom.x, V0hom.y, V0hom.z, 0.0f);

	if (p1 == p2 || p2 == p3 || p3 == p1)
	{
		// If any of the points are the same, then return no intersection
		return -1;
	}

	glm::vec3 s; // s is a point on the plane
	if (p1.x == 0.0f && p1.y == 0.0f && p1.z == 0.0f)
		s = p2;
	else
		s = p1;

	//std::vector<glm::vec3> points;
	//points.push_back(p1);
	//points.push_back(p2);
	//points.push_back(p3);
	//glm::vec3 normal = Helper::CalculateNormals(points);
	glm::vec4 N = glm::vec4(normal, 0.0f);
	surfaceNormal = glm::normalize(multiplyMV(mesh.tranposeTranform, glm::vec4(normal,0.0)));
	//surfaceNormal.x = surNormalTemp.x; surfaceNormal.y = surNormalTemp.y; surfaceNormal.z = surNormalTemp.z;

	// If the ray is parallel to the  plane and there are either no solutions or an infinite number (line in plane).
	// Returning -1 in that case
	double dotProd = glm::dot(normal, V0hom);
	if (dotProd == 0.0)
		return -1;

	double t = Test_RayPlaneIntersect(P0hom, V0hom, s, normal);

	// No intersectionin the viewing direction if t is -ve
	if (t <= 0.0)
		return -1;
	if (abs(t) < 1e-3) // on the surface
		return -1;

	glm::vec3 p = P0hom + V0hom*(float)t;
	intersectionPoint = r.origin + r.direction*(float)t;

	/*s = area(ΔP1P2P3)
	s1 = area(ΔPP2P3) / s
	s2 = area(ΔPP3P1) / s
	s3 = area(ΔPP1P2) / s*/

	// Find area of triangle - P1-P2-P3
	glm::mat3 matrix1 = glm::mat3(p1.y, p1.z, 1,
		p2.y, p2.z, 1,
		p3.y, p3.z, 1);
	glm::mat3 matrix2 = glm::mat3(p1.z, p1.x, 1,
		p2.z, p2.x, 1,
		p3.z, p3.x, 1);
	glm::mat3 matrix3 = glm::mat3(p1.x, p1.y, 1,
		p2.x, p2.y, 1,
		p3.x, p3.y, 1);
	double sArea = 0.5*std::sqrt(std::pow(glm::determinant(matrix1),2) + std::pow(glm::determinant(matrix2),2) + std::pow(glm::determinant(matrix3),2));

	// Find area of triangle - P-P2-P3
	matrix1 = glm::mat3(p.y, p.z, 1,
		p2.y, p2.z, 1,
		p3.y, p3.z, 1);
	matrix2 = glm::mat3(p.z, p.x, 1,
		p2.z, p2.x, 1,
		p3.z, p3.x, 1);
	matrix3 = glm::mat3(p.x, p.y, 1,
		p2.x, p2.y, 1,
		p3.x, p3.y, 1);
	double s1Area = 0.5*std::sqrt(std::pow(glm::determinant(matrix1),2) + std::pow(glm::determinant(matrix2),2) + std::pow(glm::determinant(matrix3),2))/sArea;

	// Find area of triangle - P-P3-P1
	matrix1 = glm::mat3(p.y, p.z, 1,
		p3.y, p3.z, 1,
		p1.y, p1.z, 1);
	matrix2 = glm::mat3(p.z, p.x, 1,
		p3.z, p3.x, 1,
		p1.z, p1.x, 1);
	matrix3 = glm::mat3(p.x, p.y, 1,
		p3.x, p3.y, 1,
		p1.x, p1.y, 1);
	double s2Area = 0.5*std::sqrt(std::pow(glm::determinant(matrix1),2) + std::pow(glm::determinant(matrix2),2) + std::pow(glm::determinant(matrix3),2))/sArea;

	// Find area of triangle - P-P1-P2
	matrix1 = glm::mat3(p.y, p.z, 1,
		p1.y, p1.z, 1,
		p2.y, p2.z, 1);
	matrix2 = glm::mat3(p.z, p.x, 1,
		p1.z, p1.x, 1,
		p2.z, p2.x, 1);
	matrix3 = glm::mat3(p.x, p.y, 1,
		p1.x, p1.y, 1,
		p2.x, p2.y, 1);
	double s3Area = 0.5*std::sqrt(std::pow(glm::determinant(matrix1),2) + std::pow(glm::determinant(matrix2),2) + std::pow(glm::determinant(matrix3),2))/sArea;

	/*P is inside if
	0 ≤ s1 ≤ 1
	0 ≤ s2 ≤ 1
	0 ≤ s3 ≤ 1
	s1 + s2 + s3 = 1*/
	if ((s1Area >= 0.0 && s1Area <= 1.0)
		&& (s2Area >= 0.0 && s2Area <= 1.0)
		&& (s3Area >= 0.0 && s3Area <= 1.0)
		&& ((s1Area + s2Area + s3Area) >= 1.0 - 1e-3)
		&& ((s1Area + s2Area + s3Area) <= 1.0 + 1e-3))
	{
		// Point is inside the triangle
		return t;
	}

	return -1;
}

//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__  float boxIntersectionTest(glm::vec3 boxMin, glm::vec3 boxMax, staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
glm::vec3 P0 = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));

glm::vec3 V0 = multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f));

ray rt;

rt.origin = P0;

rt.direction = V0;

float xmin = -0.5, xmax = 0.5;

float ymin = -0.5, ymax = 0.5;

float zmin = -0.5, zmax = 0.5;

float tFar = 999999; //std::numeric_limits<float>::max();

float tNear = -999999;//std::numeric_limits<float>::min();

float t1, t2;

// For the X planes

if (rt.direction.x == 0)

{

// Ray is || to x-axis

// The light point should be in between the xmin and xmax bounds. Else it doesn't intersect

if (rt.origin.x < xmin || rt.origin.x > xmax)

{

return -1;

}

}

else

{

// T1 = (Xl - Xo) / Xd

t1 = (xmin - rt.origin.x)/rt.direction.x;

// T2 = (Xh - Xo) / Xd

t2 = (xmax - rt.origin.x)/rt.direction.x;

// If T1 > T2 swap (T1, T2) /* since T1 intersection with near plane */

if (t1 > t2)

{

//swap t1 and t2

double temp = t1;

t1 = t2;

t2 = temp;

}

// If T1 > Tnear set Tnear =T1 /* want largest Tnear */

if (t1 > tNear)

tNear = t1;

// If T2 < Tfar set Tfar="T2" /* want smallest Tfar */

if (t2 < tFar)

tFar = t2;

// If Tnear > Tfar box is missed so return false

if (tNear > tFar)

return -1;

// If Tfar < 0 box is behind ray return false end

if (tFar < 0)

return -1;

}

// For the Y planes

if (rt.direction.y == 0)

{

// Ray is || to y-axis

// The light point should be in between the ymin and ymax bounds. Else it doesn't intersect

if (rt.origin.y < ymin || rt.origin.y > ymax)

{

return -1;

}

}

else

{

// T1 = (Yl - Yo) / Yd

t1 = (ymin - rt.origin.y)/rt.direction.y;

// T2 = (Yh - Yo) / Yd

t2 = (ymax - rt.origin.y)/rt.direction.y;

// If T1 > T2 swap (T1, T2) /* since T1 intersection with near plane */

if (t1 > t2)

{

//swap t1 and t2

double temp = t1;

t1 = t2;

t2 = temp;

}

// If T1 > Tnear set Tnear =T1 /* want largest Tnear */

if (t1 > tNear)

tNear = t1;

// If T2 < Tfar set Tfar="T2" /* want smallest Tfar */

if (t2 < tFar)

tFar = t2;

// If Tnear > Tfar box is missed so return false

if (tNear > tFar)

return -1;

// If Tfar < 0 box is behind ray return false end

if (tFar < 0)

return -1;

}

// For the Z planes

if (rt.direction.z == 0)

{

// Ray is || to z-axis

// The light point should be in between the zmin and zmax bounds. Else it doesn't intersect

if (rt.origin.z < zmin || rt.origin.z > zmax)

{

return -1;

}

}

else

{

// T1 = (Zl - Zo) / Zd

t1 = (zmin - rt.origin.z)/rt.direction.z;

// T2 = (Zh - Zo) / Zd

t2 = (zmax - rt.origin.z)/rt.direction.z;

// If T1 > T2 swap (T1, T2) /* since T1 intersection with near plane */

if (t1 > t2)

{

//swap t1 and t2

double temp = t1;

t1 = t2;

t2 = temp;

}

// If T1 > Tnear set Tnear =T1 /* want largest Tnear */

if (t1 > tNear)

tNear = t1;

// If T2 < Tfar set Tfar="T2" /* want smallest Tfar */

if (t2 < tFar)

tFar = t2;

// If Tnear > Tfar box is missed so return false

if (tNear > tFar)

return -1;

// If Tfar < 0 box is behind ray return false end

if (tFar < 0)

return -1;

}

// Box survived all above tests, return with intersection point Tnear and exit point Tfar.

double t;

if (abs(tNear) < 1e-3)

{

if (abs(tFar) < 1e-3) // on the surface

return -1;

t = tFar;

}

else

{

t = tNear;

}

glm::vec3 p = getPointOnRayUnnormalized(rt, t);

glm::vec4 surNormalTemp = glm::vec4(0.0,0.0,0.0,0.0);

if (p.x <= xmin+(1e-3) && p.x >= xmin-(1e-3))

surNormalTemp.x = -1;

if (p.y <= ymin+(1e-3) && p.y >= ymin-(1e-3))

surNormalTemp.y = -1;

if (p.z <= zmin+(1e-3) && p.z >= zmin-(1e-3))

surNormalTemp.z = -1;

if (p.x <= xmax+(1e-3) && p.x >= xmax-(1e-3))

surNormalTemp.x = 1;

if (p.y <= ymax+(1e-3) && p.y >= ymax-(1e-3))

surNormalTemp.y = 1;

if (p.z <= zmax+(1e-3) && p.z >= zmax-(1e-3))

surNormalTemp.z = 1;

normal = multiplyMV(box.tranposeTranform, surNormalTemp);

normal = glm::normalize(normal);

intersectionPoint = getPointOnRay(r, t);

//intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, t), 1.0));

return t;
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

    thrust::default_random_engine rng(hash1(randomSeed));
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
  thrust::default_random_engine rng(hash1(randomSeed));
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