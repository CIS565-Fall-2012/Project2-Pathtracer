-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Sunday, 10/12/2012
-------------------------------------------------------------------------------

BLOG Link: http://seunghoon-cis565.blogspot.com/2012/10/project-2-cuda-pathtracer.html

-------------------------------------------------------------------------------
A brief description
-------------------------------------------------------------------------------
The goal of this project is to implement a simple PathTracing algorithm by using CUDA.

-------------------------------------------------------------------------------
Features
-------------------------------------------------------------------------------
- Basic
* Full global illumination (including soft shadows, color bleeding, etc.)
* Properly accumulating emittance and colors to generate a final image
* Supersampled antialiasing
* Parallelization by ray instead of by pixel via stream compaction
* Perfect specular reflection
  


- Addtional
* Interactive camera via keyboard and mouse
* Depth of field

-------------------------------------------------------------------------------
How to build
-------------------------------------------------------------------------------
I developed this project on Visual Studio 2010.
Its solution file is located in "PROJ1_WIN/565Raytracer.sln".
You should be able to build it without modification.