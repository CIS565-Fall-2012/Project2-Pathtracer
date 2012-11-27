// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"
//Define Focal Distance Here
#define FOCALDISTANCE 8.0f
//This number defines the interval of frames between image file writes. For example, 100 means every 100 iterations there will be a file write.
#define WriteToFileFrames 100

#define Print_Cuda_Properties
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

  #ifdef __APPLE__
	  // Needed in OSX to force use of OpenGL3.2 
	  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
	  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
	  glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	  glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  #endif

	//////////////////////////////////////////CUDA DEVICE PROPERTIES///////////////////////////////////////////
#ifdef Print_Cuda_Properties
	  cudaDeviceProp  prop;

	int count;

	cudaGetDeviceCount( &count );
	printf("This machine has %d CUDA devices availiable for harvesting \n\n", count);
	for (int i=0; i< count; i++) {

	cudaGetDeviceProperties( &prop, i );
	printf( "   --- General Information for device %d ---\n", i );
	printf( "Name:  %s\n", prop.name );
	printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
	printf( "Clock rate:  %d\n", prop.clockRate );
	printf( "Device copy overlap:  " );
	if (prop.deviceOverlap)
		printf( "Enabled\n" );
	else
		printf( "Disabled\n");
	printf( "Kernel execution timeout :  " );
	if (prop.kernelExecTimeoutEnabled)
	{
		printf( "Enabled\n" );
	}
	else
		printf( "Disabled\n" );

	printf( "   --- Memory Information for device %d ---\n", i );
	printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
	printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
	printf( "Max mem pitch:  %ld\n", prop.memPitch );
	printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

	printf( "   --- MP Information for device %d ---\n", i );
	printf( "Multiprocessor count:  %d\n",
		prop.multiProcessorCount );
	printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
	printf( "Registers per mp:  %d\n", prop.regsPerBlock );
	printf( "Threads in warp:  %d\n", prop.warpSize );
	printf( "Max threads per block:  %d\n",
		prop.maxThreadsPerBlock );
	printf( "Max thread dimensions:  (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1],
		prop.maxThreadsDim[2] );
	printf( "Max grid dimensions:  (%d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1],
		prop.maxGridSize[2] );
	printf( "\n" );
	}
#endif
	//////////////////////////////////////////////////////////////////////////////////////////////////
  // Set up pathtracer stuff
  bool loadedScene = false;
  finishedRender = false;

  targetFrame = 0;
  singleFrameMode = false;

  /////////////////!!!!!SET THE INITIAL FOCAL PLANE HERE!!!!!/////////////////
  //FocusPoint = glm::vec3(0.0, 0.0, 0.0);
  FocalDistance = FOCALDISTANCE;

  // Load scene file
  for(int i=1; i<argc; i++){
	string header; string data;
	istringstream liness(argv[i]);
	getline(liness, header, '='); getline(liness, data, '=');
	if(strcmp(header.c_str(), "scene")==0){
	  renderScene = new scene(data);
	  loadedScene = true;
	}else if(strcmp(header.c_str(), "frame")==0){
	  targetFrame = atoi(data.c_str());
	  singleFrameMode = true;
	}
  }

  if(!loadedScene){
	cout << "Error: scene file needed!" << endl;
	return 0;
  }

  // Set up camera stuff from loaded pathtracer settings
  iterations = 0;
  renderCam = &renderScene->renderCam;
  width = renderCam->resolution[0];
  height = renderCam->resolution[1];
   for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
		renderCam->image[i] = glm::vec3(1.0,1.0,1.0);
	  }
   

  if(targetFrame>=renderCam->frames){
	cout << "Warning: Specified target frame is out of range, defaulting to frame 0." << endl;
	targetFrame = 0;
  }
  
  //*OriginalCamera = renderScene->renderCam;
  OriginalCameraPosition = renderCam->positions[0];
  OriginalCameraView = renderCam->views[0];
  // Launch CUDA/GL

  #ifdef __APPLE__
	init();
  #else
	init(argc, argv);
  #endif

  initCuda();

  initVAO();
  initTextures();

  GLuint passthroughProgram;
  passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  #ifdef __APPLE__
	  // send into GLFW main loop
	  while(1){
		display();
		if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS || !glfwGetWindowParam( GLFW_OPENED )){
				exit(0);
		}
	  }

	  glfwTerminate();
  #else
	  glutDisplayFunc(display);
	  glutKeyboardFunc(keyboard);

	  glutMainLoop();
  #endif
  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){

  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
  if(changed == true)
  {
	  iterations = 0;
	  changed = false;
  }
  if(iterations<renderCam->iterations){
	uchar4 *dptr=NULL;
	
	iterations++;
	cudaGLMapBufferObject((void**)&dptr, pbo);
  
	//pack geom and material arrays
	geom* geoms = new geom[renderScene->objects.size()];
	material* materials = new material[renderScene->materials.size()];
	
	for(int i=0; i<renderScene->objects.size(); i++){
	  geoms[i] = renderScene->objects[i];
	}
	for(int i=0; i<renderScene->materials.size(); i++){
	  materials[i] = renderScene->materials[i];
	}
	
	
	// execute the kernel
	cudaRaytraceCore(dptr, renderCam, targetFrame, iterations, materials, renderScene->materials.size(), geoms, renderScene->objects.size(), changed, FocalDistance);
#ifdef WriteToFileFrames
	if(WriteToFileFrames != 0)
	{
		if(iterations % WriteToFileFrames == 0)
			WriteToFile();
	}
#endif

	// unmap buffer object
	cudaGLUnmapBufferObject(pbo);
  }else{

	if(!finishedRender){
	  //output image file
	  image outputImage(renderCam->resolution.x, renderCam->resolution.y);

	  for(int x=0; x<renderCam->resolution.x; x++){
		for(int y=0; y<renderCam->resolution.y; y++){
		  int index = x + (y * renderCam->resolution.x);
		  outputImage.writePixelRGB(x,y,renderCam->image[index]);
		}
	  }
	  
	  gammaSettings gamma;
	  gamma.applyGamma = true;
	  gamma.gamma = 1.0/2.2;
	  gamma.divisor = renderCam->iterations;
	  outputImage.setGammaSettings(gamma);
	  string filename = renderCam->imageName;
	  string s;
	  stringstream out;
	  out << targetFrame;
	  s = out.str();
	  utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
	  utilityCore::replaceString(filename, ".png", "."+s+".png");
	  outputImage.saveImageRGB(filename);
	  cout << "Saved frame " << s << " to " << filename << endl;
	  finishedRender = true;
	  if(singleFrameMode==true){
		cudaDeviceReset(); 
		exit(0);
	  }
	}
	if(targetFrame<renderCam->frames-1){

	  //clear image buffer and move onto next frame
	  targetFrame++;
	  iterations = 0;
	  for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
		renderCam->image[i] = glm::vec3(0,0,0);
	  }
	  cudaDeviceReset(); 
	  finishedRender = false;
	}
  }
  
}

#ifdef __APPLE__

	void display(){
		runCuda();

		string title = "CIS565 Render | " + utilityCore::convertIntToString(iterations) + " Frames";
		glfwSetWindowTitle(title.c_str());

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glClear(GL_COLOR_BUFFER_BIT);   

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		glfwSwapBuffers();
	}

#else

	void display(){
		runCuda();

		string title = "565Raytracer | " + utilityCore::convertIntToString(iterations) + " Frames";
		glutSetWindowTitle(title.c_str());

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glClear(GL_COLOR_BUFFER_BIT);   

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		glutPostRedisplay();
		glutSwapBuffers();
	}

	void keyboard(unsigned char key, int x, int y)
	{
		std::cout << key << std::endl;
		float radAngle;
		glm::mat3 RotateX;
		switch (key) 
		{
		   case(27):
			   exit(1);
			   break;

		   case 'w':
			   renderCam->positions[0].y += 0.5f;
			   renderCam->positions[1].y += 0.5f;
			   changed = true;
			   break;

			case 's':
			   renderCam->positions[0].y -= 0.5f;
			   renderCam->positions[1].y -= 0.5f;
			   changed = true;
			   break;

			case 'd':
			   renderCam->positions[0].x += 0.5f;
			   renderCam->positions[1].x += 0.5f;
			   changed = true;
			   break;

			case 'a':
			   renderCam->positions[0].x -= 0.5f;
			   renderCam->positions[1].x -= 0.5f;
			   changed = true;
			   break;

			case 'q':
			   renderCam->positions[0].z += 0.5f;
			   renderCam->positions[1].z += 0.5f;
			   changed = true;
			   break;

			case 'e':
			   renderCam->positions[0].z -= 0.5f;
			   renderCam->positions[1].z -= 0.5f;
			   changed = true;
			   break;

			case 'x':
				radAngle = glm::radians(0.2);
				RotateX = glm::mat3(1.0, 0.0, 0.0, 0.0, cos(radAngle),  -sin(radAngle), 0.0, sin(radAngle), cos(radAngle));
				renderCam->views[0] = RotateX * renderCam->views[0];
				renderCam->views[1] = RotateX * renderCam->views[1];
				changed = true;
				break;

			case 'X':
				radAngle = glm::radians(-0.2);
				RotateX = glm::mat3(1.0, 0.0, 0.0, 0.0, cos(radAngle),  -sin(radAngle), 0.0, sin(radAngle), cos(radAngle));
				renderCam->views[0] = RotateX * renderCam->views[0];
				renderCam->views[1] = RotateX * renderCam->views[1];
				changed = true;
				break;

			case 'y':
				radAngle = glm::radians(0.2);
				RotateX = glm::mat3(cos(radAngle), 0.0, sin(radAngle), 0.0, 1.0, 0.0, -sin(radAngle), 0.0, cos(radAngle));
				renderCam->views[0] = RotateX * renderCam->views[0];
				renderCam->views[1] = RotateX * renderCam->views[1];
				changed = true;
				break;

			case 'Y':
				radAngle = glm::radians(-0.2);
				RotateX = glm::mat3(cos(radAngle), 0.0, sin(radAngle), 0.0, 1.0, 0.0, -sin(radAngle), 0.0, cos(radAngle));
				renderCam->views[0] = RotateX * renderCam->views[0];
				renderCam->views[1] = RotateX * renderCam->views[1];
				changed = true;
				break;

			///*case 'y':
			//	radAngle = glm::radians(0.2);
			//	RotateX = glm::mat3(cos(radAngle), -sin(radAngle), 0.0, sin(radAngle), cos(radAngle), 0.0, 0.0, 0.0, 1.0);
			//	renderCam->views[0] = RotateX * renderCam->views[0];
			//	renderCam->views[1] = RotateX * renderCam->views[1];
			//	break;

			//case 'Y':
			//	radAngle = glm::radians(-90.0);
			//	RotateX = glm::mat3(cos(radAngle), -sin(radAngle), 0.0, sin(radAngle), cos(radAngle), 0.0, 0.0, 0.0, 1.0);
			//	renderCam->views[0] = RotateX * renderCam->views[0];
			//	renderCam->views[1] = RotateX * renderCam->views[1];
			//	renderCam->ups[0] = RotateX * renderCam->views[0];
			//	renderCam->ups[1] = RotateX * renderCam->views[1];
			//	break;*/
			
			case 'i':		//Write image to file
			case 'I':
				WriteToFile();
				break;

			case 'j':
				//FocusPoint += glm::vec3(0.0f, 0.0f, 0.5f);
				FocalDistance += 0.5;
				changed = true;
				break;

			case 'k':
				//FocusPoint += glm::vec3(0.0f, 0.0f, -0.5f);
				FocalDistance -= 0.5;
				changed = true;
				break;

			case '0':
				renderCam->positions[0] = OriginalCameraPosition;
				renderCam->views[0] = OriginalCameraView;
				changed = true;
				break;

			case 'p':
				std::cout << "Paused - Press 1 and then hit Enter To Continue...." << std::endl;
				int c;
				std::cin >> c;
				if(c < 0)
					exit(1);
				break;
		}
	}

#endif



//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
	void init(){

		if (glfwInit() != GL_TRUE){
			shut_down(1);      
		}

		// 16 bit color, no depth, alpha or stencil buffers, windowed
		if (glfwOpenWindow(width, height, 5, 6, 5, 0, 0, 0, GLFW_WINDOW) != GL_TRUE){
			shut_down(1);
		}

		// Set up vertex array object, texture stuff
		initVAO();
		initTextures();
	}
#else
	void init(int argc, char* argv[]){
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(width, height);
		glutCreateWindow("565Raytracer");

		// Init GLEW
		glewInit();
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			/* Problem: glewInit failed, something is seriously wrong. */
			std::cout << "glewInit failed, aborting." << std::endl;
			exit (1);
		}

		initVAO();
		initTextures();
	}
#endif

void initPBO(GLuint* pbo){
  if (pbo) {
	// set up vertex data parameter
	int num_texels = width*height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;
	
	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1,pbo);
	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject( *pbo );
  }
}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

  initPBO(&pbo);

  // Clean up on program exit
  atexit(cleanupCuda);

  runCuda();
}

void initTextures(){
	glGenTextures(1,&displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
	GLfloat vertices[] =
	{ 
		-1.0f, -1.0f, 
		 1.0f, -1.0f, 
		 1.0f,  1.0f, 
		-1.0f,  1.0f, 
	};

	GLfloat texcoords[] = 
	{ 
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);
	
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
	GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
	GLint location;

	glUseProgram(program);
	
	if ((location = glGetUniformLocation(program, "u_image")) != -1)
	{
		glUniform1i(location, 0);
	}

	return program;
}

//-------------------------------
//-----Write Render To File------
//-------------------------------
void WriteToFile()
{
	image outputImage(renderCam->resolution.x, renderCam->resolution.y);

	for(int x=0; x<renderCam->resolution.x; x++)
	{
		for(int y=0; y<renderCam->resolution.y; y++)
		{
			int index = x + (y * renderCam->resolution.x);
			outputImage.writePixelRGB(renderCam->resolution.x - x - 1,y,renderCam->image[index]);
		}
	}

	string filename = renderCam->imageName;
	string s;
	stringstream out;
	out << iterations;
	s = out.str();
	utilityCore::replaceString(filename, ".bmp", "_"+s+".bmp");
	utilityCore::replaceString(filename, ".png", "_"+s+".png");
	outputImage.saveImageRGB(filename);
	cout << "Saved frame " << s << " to " << filename << endl;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
	// unregister this buffer object with CUDA
	cudaGLUnregisterBufferObject(*pbo);
	
	glBindBuffer(GL_ARRAY_BUFFER, *pbo);
	glDeleteBuffers(1, pbo);
	
	*pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
  #ifdef __APPLE__
	glfwTerminate();
  #endif
  exit(return_code);
}
