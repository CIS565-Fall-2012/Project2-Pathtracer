-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Friday, 10/12/2012
-------------------------------------------------------------------------------
PROJECT SUBMISSION
--------------------------------------------------------------------------------------------------------------------------------------------------------------
The features included in my GPU-based Parallel Path Tracer are as follows:

Ray Parallelization
	Instead of working on parallelizing by pixel, I working on this project by parallelizing rays. That is, instead of assign 1 thread per pixel, I assigned 1 thread per ray. This became fruitful in the scenes where there was openness. Since most rays would be terminated early, in the later iterations I saved compute space by not having to call the rays which were dead. This was done by using stream compaction using the thrust::remove_if() function.

Global Illumination
	Parallel path tracing makes global illumination really easy. Since we are averaging a large number of iterations, global illumination by distributed random sampling becomes very easy and very efficient. All you have to take care of is the terminating conditions and you BSDF (well this can be a little tricky). My approach to BSDF was to start of simple - handle only diffuse objects, then move to reflective, refractive, glossy etc. Building up from a simple model BSDF is very crucial to understanding how it works.
	My BDSF function handles diffuse, reflective, refractive, glossy, transparent, and translucent surfaces.
	It produces caustics, soft shadows, color bleeding effects.
	I have also used a very basic form of tone mapping in some of my images.

Super-sampled Anti-aliasing
	The advantage of working on a path tracer on a GPU is that you have loads of compute power at your disposal. Implementing super-sampled anti-aliasing was a piece of cake given this advantage. For each ray that you begin, just jitter it a little before you shoot the ray. Since we are going to iterating over a lot of frames, averaging the values makes it extremely good for anti-aliasing because we can get infinite number of jittered points.

Depth of Field
	Firstly, I must thank my friend classmate, Tiju Thomas, for helping understand how to implement this. 
	For depth of field, I use a Focal Distance. All object at this focal distance from the camera are in focus whereas other objects are blurred.
	The algorithm used is as follows:
		->Compute the ray direction on the image plane.
		->Compute the point on this ray at focal distance.
		->Jitter the camera based on aperture (preferably perpendicular to the view direction).
		->New origin of the ray is the jittered camera position.
		->New direction of the ray is the Focal Point (from Step 2) - Jittered camera position from step 3.
	Its amazing how such a simple mechanism can produce brilliant effects.
	I also implemented keyboard inputs to increase or decrease the focal distance.

Fresnel Refraction
	I used Schlick's approximation to compute Fresnel coefficients. To all those who think Fresnel is hard (I used to be one of them), the best way to get around is to implement a perfect Snell Law refraction mechanism. Once this is done Fresnel is a piece of cake. 
	Use Schlick's approximation to compute the reflection and refraction coefficients. Use a random number to choose reflection or refraction. If its reflection, do regular reflection. If its refraction then  do refraction or reflection (based on critical angle).

You can find my blog at: mzshehzanayub.blogspot.com
I'll keep updating my blog as I add more features to it. Sometime in the near future, I will definitely add a OBJ loader.

I also plan to make a tutorial video that will explain all concepts I have used in my path tracer.
-------------------------------------------------------------------------------
HOW TO MAKE IT WORK
-------------------------------------------------------------------------------
1. Enter the path of the scene file as argument.

2. Tags (Using #define):
	In main.cpp
	a. Line 10: #define FOCALDISTANCE <Float Value>
		- This #define is used to set the distance of the focal plane from the camera. (Note: Setting this makes no difference if you do not enable #define USE_DEPTH_OF_FEILD from Line 40 of rayTraceKernel.cu.
	b. Line 12: #define Write_To_File_Frames <int value>
		- This #define define the interval between image file writes. Setting it to 0 (or commenting it) will disable image file write.
	
	In raytraceKernel.cu
	a. Line 33: #define MAX_DEPTH <int value>
		- This value define the Max Depth of each iteration. By default, its set at 8.
	b. Line 36: #define StreamCompactDepth <int value>
		- This value is used to decide the depths at which stream compact will be run. For eg,if it is 3, then stream compaction will run every 3rd depth. Setting it to 0 will turn stream compaction off. By defualt, its set to 3. This is because I noted that stream compaction slows it down in reasonably closed scenes, so it is better to run it at intervals.
	c. Line 40: #define USE_DEPTH_OF_FIELD	
		- Comment this line to turn depth of field off. Uncomment the line to turn depth of field on. By defualt, its uncommented (depth of feild enabled).
	d. Line 44: #define USE_ANTI_ALIASING
		- Comment this line to turn off Anti-aliasing. Uncomment the line to turn on Anti-aliasing. By defualt, its uncommented (anti-aliasing enabled).
		
	In interations.h
	a. Line 09: #define USE_FRESNEL_OR_SNELL
		- Comment for SNELL, Uncomment for Fresnel. By defualt, its uncommented (set to Fresnel).
		
3. Run it.

4. User Interaction (Keyboard - case sensitive unless mentioned):
	case 'w': move the camera +0.5 along Y-axis
	
	case 's': move the camera -0.5 along Y-axis

	case 'd': move the camera +0.5 along X-axis
	
	case 'a': move the camera -0.5 along X-axis

	case 'q': move the camera +0.5 along Z-axis

	case 'e': move the camera -0.5 along Z-axis

	case 'x': rotate about X-axis in anti-clockwise direction
				
	case 'X': rotate about X-axis in clockwise direction

	case 'y': rotate about Y-axis in anti-clockwise direction
				
	case 'Y': rotate about Y-axis in clockwise direction
			
	case 'i': Write current image frame to file (Key is case insensitive)
			
	case 'j': Increase Focal Distance by 0.5.

	case 'k': Decrease Focal Distance by 0.5.

	case '0': Reset the camera to the original position as defined in the scene file.
	
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