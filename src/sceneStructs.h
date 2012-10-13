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
#include <vector>

enum GEOMTYPE{ SPHERE, CUBE, MESH };

struct ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

struct Triangle
{
	unsigned int p1;
	unsigned int p2;
	unsigned int p3;

	Triangle()
	{
		p1 = 0;
		p2 = 0;
		p3 = 0;
	}

	Triangle(unsigned int m_p1, unsigned int m_p2, unsigned int m_p3)
	{
		p1 = m_p1;
		p2 = m_p2;
		p3 = m_p3;
	}
};

struct Mesh
{
	std::vector<glm::vec3> vertices;
	std::vector<Triangle> faces;
	std::vector<glm::vec3> normals;

	// Calculate the normals for the passed list of vertices of a polygon
	// The last vertex will be duplicated as the end vertex, if it is not already present
	glm::vec3 CalculateNormals(unsigned int faceIndex)
	{
		std::vector<glm::vec3> vertexList;
		vertexList.push_back(vertices[faces[faceIndex].p1]);
		vertexList.push_back(vertices[faces[faceIndex].p2]);
		vertexList.push_back(vertices[faces[faceIndex].p3]);
		vertexList.push_back(vertices[faces[faceIndex].p1]);

		float a=0.0f, b=0.0f, c=0.0f;
		for(unsigned int i=0; i<(vertexList.size()-1); ++i)
		{
			a+=((vertexList[i].y - vertexList[i+1].y) * (vertexList[i].z + vertexList[i+1].z));
			b+=((vertexList[i].z - vertexList[i+1].z) * (vertexList[i].x + vertexList[i+1].x));
			c+=((vertexList[i].x - vertexList[i+1].x) * (vertexList[i].y + vertexList[i+1].y));
		}

		return glm::normalize(glm::vec3(a,b,c));
	}
};

struct MeshCuda
{
	glm::vec3* vertices;
	unsigned int numVertices;
	glm::vec3* normals;
	unsigned int numNormals;
	Triangle* faces;
	unsigned int numFaces;

	MeshCuda()
	{
		vertices = NULL;
		faces = NULL;
		normals = NULL;
		numVertices = 0;
		numFaces = 0;
		numNormals = 0;
	}

	MeshCuda(glm::vec3* p_vertices, unsigned int p_numVertices, glm::vec3* p_normals, unsigned int p_numNormals, 
		     Triangle* p_faces, unsigned int p_numFaces)
	{
		cudaMalloc((void**)&vertices, p_numVertices*sizeof(glm::vec3));
		cudaMemcpy(vertices, p_vertices, p_numVertices*sizeof(glm::vec3), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&faces, p_numFaces*sizeof(Triangle));
		cudaMemcpy(faces, p_faces, p_numFaces*sizeof(Triangle), cudaMemcpyHostToDevice);
		
		cudaMalloc((void**)&normals, p_numNormals*sizeof(glm::vec3));
		cudaMemcpy(normals, p_normals, p_numNormals*sizeof(glm::vec3), cudaMemcpyHostToDevice);
		
		numVertices = p_numVertices;
		numFaces = p_numFaces;
		numNormals = p_numNormals;
	}

	~MeshCuda()
	{
		cudaFree(vertices);
		cudaFree(faces);
		cudaFree(normals);
	}
};

struct geom {
	enum GEOMTYPE type;
	int meshIndex;
	int materialid;
	int frames;
	glm::vec3* translations;
	glm::vec3* rotations;
	glm::vec3* scales;
	cudaMat4* transforms;
	cudaMat4* inverseTransforms;
	cudaMat4* inverseTransposeTransforms;
};

struct staticGeom {
	enum GEOMTYPE type;
	int meshIndex;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
	cudaMat4 inverseTransposeTransform;
};

struct cameraData {
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
	float aperture;
	float focalDist;
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
	float aperture;
	float focalDistance;
};

struct material{
	glm::vec3 color;
	float specularExponent;
	glm::vec3 specularColor;
	float reflectivity;
	bool hasRefractive;
	float indexOfRefraction;
	float hasScatter;
	glm::vec3 absorptionCoefficient;
	float reducedScatterCoefficient;
	float emittance;
};

class RayPackage
{
public:
	ray rayObj;
	int index;
	glm::vec3 color;
	bool inside;
	bool isPresent;

public:
	RayPackage()
	{
		rayObj.origin = glm::vec3(0,0,0);
		rayObj.direction = glm::vec3(0,0,0);
		color = glm::vec3(0,0,0);
		inside = false;
		isPresent = false;
	}


	RayPackage(ray m_rayObj, glm::vec3 m_color, bool m_inside, bool m_isPresent)
	{
		rayObj.origin = m_rayObj.origin;
		rayObj.direction = m_rayObj.direction;
		color = m_color;
		inside = m_inside;
		isPresent = m_isPresent;
	}

	void SetValues(ray m_rayObj, glm::vec3 m_color, bool m_inside, bool m_isPresent)
	{
		rayObj.origin = m_rayObj.origin;
		rayObj.direction = m_rayObj.direction;
		color = m_color;
		inside = m_inside;
		isPresent = m_isPresent;
	}
};

#endif //CUDASTRUCTS_H
