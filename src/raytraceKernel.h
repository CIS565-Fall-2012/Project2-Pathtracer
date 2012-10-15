// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef RAYTRACEKERNEL_H
#define RAYTRACEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>

#define RAYTRACEKERNEL_RAY_BOUNCE_MAX 10
#define RAYTRACEKERNEL_DEPTH_OF_FIELD 

typedef struct
{
  ray r;
  
  // pixel index in 1D
  // traversing 2D image plane in the row-major order
  // negative value means this ray is no longer considered.  
  int pixelID; 
} PixelRay;

struct isTerminated
{
  __host__ __device__
  bool operator()(const PixelRay pixRay)
  {
    return pixRay.pixelID < 0;
  }
};

void cudaRaytraceCore(uchar4* pos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms);

// kernel to prepare some data structures needed by PathTracing
__global__ void preparePathTracing(const cameraData* cam, float time,
                                 PixelRay* pixelRays, glm::vec3* acc_refl_diff_colors);

//// kernel to accumulate colors acheived in this iteration to the total accumulator
//__global__ void accumulateColors(const glm::vec2 resolution, glm::vec3* totalAccumulator, const glm::vec3* accuThisIter);

#endif
