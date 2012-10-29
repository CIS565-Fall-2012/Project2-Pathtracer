// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef CUDASTRUCTS_H
#define CUDASTRUCTS_H

#include "glm/glm.hpp"
#include "cudaMat4.h"
#include <cuda_runtime.h>
#include <string>

enum GEOMTYPE{ SPHERE, CUBE, MESH };

struct ray {
	glm::vec3 origin;
	glm::vec3 direction;
	bool keep;
	int pixelIndex;
	bool isInside;
	int currentObjIndex;
	int currentMatIndex;
};

struct TriangleStruct
{
	int index;
	glm::vec3 vertices[3];
	glm::vec3 normal[3];
};

struct geom {
	enum GEOMTYPE type;
	int materialid;
	int frames;
	glm::vec3* translations;
	glm::vec3* rotations;
	glm::vec3* scales;
	cudaMat4* transforms;
	cudaMat4* inverseTransforms;
	TriangleStruct* triangles;
	int numOfTriangles;
	//For Bounding Sphere
	cudaMat4  BSTransform;
	cudaMat4  BSInverseTransform;
	glm::vec3 BStranslation;
	glm::vec3 BSrotation;
	glm::vec3 BSscale;
};

struct staticGeom {
	enum GEOMTYPE type;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
	TriangleStruct* triangles;
	int numOfTriangles;
	//For Bounding Sphere
	cudaMat4  BSTransform;
	cudaMat4  BSInverseTransform;
	glm::vec3 BStranslation;
	glm::vec3 BSrotation;
	glm::vec3 BSscale;
};

struct cameraData {
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
};

struct camera {
	glm::vec2 resolution;
	glm::vec3* positions;
	glm::vec3* views;
	glm::vec3* ups;
	int frames;
	glm::vec2 fov;
	unsigned int iterations;
	glm::vec3* image;
	ray* rayList;
	std::string imageName;
};

struct material{
	glm::vec3 color;
	float specularExponent;
	glm::vec3 specularColor;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float hasScatter;
	glm::vec3 absorptionCoefficient;
	float reducedScatterCoefficient;
	float emittance;
};

struct intersectionStruct
{
	bool check;
	glm::vec3 normal;
	glm::vec3 intersectionPoint;
	int closestIndex;
};

struct is_alive
{
	__host__ __device__	bool operator()(ray r)
	{
		return r.keep == 0;
	}
};

#endif //CUDASTRUCTS_H
