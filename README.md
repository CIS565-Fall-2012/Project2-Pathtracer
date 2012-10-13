Assignment Submission

Funtionality implemented:
* Full global illumination (including soft shadows, color bleeding, etc.) by pathtracing rays through the scene. 
* Properly accumulating emittance and colors to generate a final image
* Supersampled antialiasing - working on it
* Parallelization by ray instead of by pixel via string compaction (see the Physically-based shading and pathtracing lecture slides from 09/24 if you don't know what this refers to)
* Perfect specular reflection

Additional Features implemented:
* Fresnel-based Refraction, i.e. glass
* Depth of field - working on it

Blog: http://cudapathtacer.blogspot.com/




-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Friday, 10/12/2012
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
This project requires an NVIDIA graphics card with CUDA capability! Any card after the Geforce 8xxx series will work. If you do not have an NVIDIA graphics card in the machine you are working on, feel free to use any machine in the SIG Lab or in Moore100 labs. All machines in the SIG Lab and Moore100 are equipped with CUDA capable NVIDIA graphics cards. If this too proves to be a problem, please contact Patrick or Karl as soon as possible.

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
In this project, you will extend your raytracer from Project 1 into a full CUDA based global illumination pathtracer. 

For this project, you may either choose to continue working off of your codebase from Project 1, or you may choose to use the included basecode in this repository. The basecode for Project 2 is the same as the basecode for Project 1, but with some missing components you will need filled in, such as the intersection testing and camera raycasting methods. 

How you choose to extend your raytracer into a pathtracer is a fairly open-ended problem; the supplied basecode is meant to serve as one possible set of guidelines for doing so, but you may choose any approach you want in your actual implementation, including completely scrapping the provided basecode in favor of your own from-scratch solution.

-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project2 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio solution and the OSX makefile reference this folder for all source; the base source code compiles on OSX and Windows without modification.
* scenes/ contains an example scene description file.
* renders/ contains two example renders: the raytraced render from Project 1 (GI_no.bmp), and the same scene rendered with global illumination (GI_yes.bmp). 
* PROJ1_WIN/ contains a Windows Visual Studio 2010 project and all dependencies needed for building and running on Windows 7.
* PROJ1_OSX/ contains a OSX makefile, run script, and all dependencies needed for building and running on Mac OSX 10.8. 

The Windows and OSX versions of the project build and run exactly the same way as in Project0 and Project1.

-------------------------------------------------------------------------------
REQUIREMENTS:
-------------------------------------------------------------------------------
In this project, you are given code for:

* All of the basecode from Project 1, plus:
* Intersection testing code for spheres and cubes
* Code for raycasting from the camera

You will need to implement the following features. A number of these required features you may have already implemented in Project 1. If you have, you are ahead of the curve and have less work to do! 

* Full global illumination (including soft shadows, color bleeding, etc.) by pathtracing rays through the scene. 
* Properly accumulating emittance and colors to generate a final image
* Supersampled antialiasing
* Parallelization by ray instead of by pixel via string compaction (see the Physically-based shading and pathtracing lecture slides from 09/24 if you don't know what this refers to)
* Perfect specular reflection

You are also required to implement at least two of the following features. Some of these features you may have already implemented in Project 1. If you have, you may NOT resubmit those features and instead must pick two new ones to implement.

* Additional BRDF models, such as Cook-Torrance, Ward, etc. Each BRDF model may count as a separate feature. 
* Texture mapping 
* Bump mapping
* Translational motion blur
* Fresnel-based Refraction, i.e. glass
* OBJ Mesh loading and rendering without KD-Tree
* Interactive camera
* Integrate an existing stackless KD-Tree library, such as CUKD (https://github.com/unvirtual/cukd)
* Depth of field

Alternatively, implementing just one of the following features can satisfy the "pick two" feature requirement, since these are correspondingly more difficult problems:

* Physically based subsurface scattering and transmission
* Implement and integrate your own stackless KD-Tree from scratch. 
* Displacement mapping
* Deformational motion blur

As yet another alternative, if you have a feature or features you really want to implement that are not on this list, let us know, and we'll probably say yes!

-------------------------------------------------------------------------------
NOTES ON GLM:
-------------------------------------------------------------------------------
This project uses GLM, the GL Math library, for linear algebra. You need to know two important points on how GLM is used in this project:

* In this project, indices in GLM vectors (such as vec3, vec4), are accessed via swizzling. So, instead of v[0], v.x is used, and instead of v[1], v.y is used, and so on and so forth.
* GLM Matrix operations work fine on NVIDIA Fermi cards and later, but pre-Fermi cards do not play nice with GLM matrices. As such, in this project, GLM matrices are replaced with a custom matrix struct, called a cudaMat4, found in cudaMat4.h. A custom function for multiplying glm::vec4s and cudaMat4s is provided as multiplyMV() in intersections.h.

-------------------------------------------------------------------------------
BLOG
-------------------------------------------------------------------------------
As mentioned in class, all students should have student blogs detailing progress on projects. If you already have a blog, you can use it; otherwise, please create a blog using www.blogger.com or any other tool, such as www.wordpress.org. Blog posts on your project are due on the SAME DAY as the project, and should include:

* A brief description of the project and the specific features you implemented.
* A link to your github repo if the code is open source.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.  To create the video use http://www.microsoft.com/expression/products/Encoder4_Overview.aspx 

-------------------------------------------------------------------------------
THIRD PARTY CODE POLICY
-------------------------------------------------------------------------------
* Use of any third-party code must be approved by asking on Piazza.  If it is approved, all students are welcome to use it.  Generally, we approve use of third-party code that is not a core part of the project.  For example, for the ray tracer, we would approve using a third-party library for loading models, but would not approve copying and pasting a CUDA function for doing refraction.
* Third-party code must be credited in README.md.
* Using third-party code without its approval, including using another student's code, is an academic integrity violation, and will result in you receiving an F for the semester.

-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------
* On the submission date, email your grade, on a scale of 0 to 100, to Karl, yiningli@seas.upenn.edu, with a one paragraph explanation.  Be concise and realistic.  Recall that we reserve 30 points as a sanity check to adjust your grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We hope to only use this in extreme cases when your grade does not realistically reflect your work - it is either too high or too low.  In most cases, we plan to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as the path tracer.  We will determine the weighting at the end of the semester based on the size of each project.

-------------------------------------------------------------------------------
SUBMISSION
-------------------------------------------------------------------------------
As with the previous project, you should fork this project and work inside of your fork. Upon completion, commit your finished project back to your fork, and make a pull request to the master repository.
You should include a README.md file in the root directory detailing the following

* A brief description of the project and specific features you implemented
* At least one screenshot of your project running, and at least one screenshot of the final rendered output of your pathtracer
* Instructions for building and running your project if they differ from the base code
* A link to your blog post detailing the project
* A list of all third-party code used