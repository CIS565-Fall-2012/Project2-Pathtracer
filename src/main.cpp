// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li and Gundeep Singh, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       //Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer:
#include "main.h"

//////////////////////////////////
//////////Interactive camera/////
/////////////////////////////////

void fixYaw() {
	renderCam->yaw = glm::mod(renderCam->yaw,6.28f); // Normalize the yaw.
}

void fixPitch() {
	float padding = 0.05;
	renderCam->pitch = clamp(renderCam->pitch, -PI/2 + padding, PI/2 - padding); // Limit the pitch.
}

void fixRadius() {
	float minRadius = 0.02;
	float maxRadius = 100.0;
	renderCam->radius = clamp(renderCam->radius, minRadius, maxRadius);
}
void fixApertureRadius() {
	float minApertureRadius = 0.0;
	float maxApertureRadius = 25.0;
	renderCam->apertureRadius = clamp(renderCam->apertureRadius, minApertureRadius, maxApertureRadius);
}

void changeYaw(float m){
	renderCam->yaw += m;
	fixYaw();
}

void changePitch(float m){
	renderCam->pitch += m;
	fixPitch();
}

void changeRadius(float m){
	renderCam->radius += renderCam->radius * m; // Change proportional to current radius. Assuming radius isn't allowed to go to zero.
	//fixRadius();
}

void changeAltitude(float m){
	renderCam->centerPosition.y += m;
	//fixCenterPosition();
}

void changeApertureDiameter(float m){
	renderCam->apertureRadius += (renderCam->apertureRadius + 0.01) * m; // Change proportional to current apertureRadius.
	//fixApertureRadius();
}

float rx;
float ry;
float dist;
const float ZOOM_STEP = 0.01f;

void zoom(float dz) {
	dist = clamp(dist - ZOOM_STEP*dz, 1.5f, 10.0f);
}

void rotate(float dx, float dy) {
	if (abs(dx) > 0.0f) {
        rx += dx;
        rx = fmod(rx,360.0f);
	}
	if (abs(dy) > 0.0f) {
        ry += dy;
        ry = clamp(ry, - (4.0f/5.0f)*90.0f, (4.0f/5.0f)*90.0f);
	}
}




int mouse_buttons = 0;
int mouse_old_x = 0;
int mouse_old_y = 0;
int theModifierState=0;

 
void motion(int x, int y)
{	
	
    float dx, dy;
    dx = (float)( mouse_old_x-x);
    dy = (float)( mouse_old_y-y);

	 if( dx !=0 || dy!=0)
	 {
		if (mouse_buttons == GLUT_RIGHT_BUTTON)  // Rotate
			{
				//cout<<"mouse left button";
				changeYaw(dx * 0.01);
				changePitch(-dy * 0.01);
			}
			else if (mouse_buttons == GLUT_MIDDLE_BUTTON) // Zoom
			{
				changeAltitude(-dy * 0.01);
			}    

			if (mouse_buttons ==  GLUT_LEFT_BUTTON) // camera move
			{
				changeRadius(-dy*0.01f);
			}
		mouse_old_x = x;
		mouse_old_y = y;
		//cudaDeviceReset();
		//pathtracerReset();
	//	deletePBO(&pbo);
		//runCuda(); 
		
		for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
		renderCam->image[i] = glm::vec3(0,0,0);
		}
		iterations=0;
		glutPostRedisplay();
	 }
}

void mouse(int button, int state, int x, int y)
{
    mouse_buttons = button;
	theModifierState = glutGetModifiers();
	mouse_old_x = x;
	mouse_old_y = y;
	motion(x, y);

}


void reshape(int w, int h)
{
	glViewport(0,0,(GLsizei)w,(GLsizei)h);
}



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

  // Set up pathtracer stuff
  bool loadedScene = false;
  finishedRender = false;

  targetFrame = 0;
  singleFrameMode = false;

  // Load scene file
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); 
	getline(liness, data, '=');
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

  if(targetFrame>=renderCam->frames){
    cout << "Warning: Specified target frame is out of range, defaulting to frame 0." << endl;
    targetFrame = 0;
  }

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

  //initSSAO();
  //initQuad();

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
		glewInit();
		GLenum err=glewInit();
		if(GLEW_OK !=err)
		{
			cout<<"glew failed"<<endl;
			exit(1);
		}


	  glutDisplayFunc(display);
	  glutReshapeFunc(reshape);	
	  glutKeyboardFunc(keyboard);
	  glutMouseFunc(mouse);
	  glutMotionFunc(motion);

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
  
  if(iterations<renderCam->iterations){
    uchar4 *dptr=NULL;
    iterations++;
    cudaGLMapBufferObject((void**)&dptr, pbo);  // hmm pbo is program buffer

	//////////////////////////////////////////  
    ///////pack geom and material arrays//////
	//////////////////////////////////////////

    geom* geoms = new geom[renderScene->objects.size()];
	obj* objs= new obj[renderScene->meshes.size()];
    material* materials = new material[renderScene->materials.size()];
    
    for(unsigned int i=0; i<renderScene->objects.size(); i++)
	{
      geoms[i] = renderScene->objects[i];
    }
	for(unsigned int k=0; k<renderScene->meshes.size(); k++)
	{
		
		objs[k]= renderScene->meshes[k];
		//objs[0].faces
		//cout<<"filling objs\n";
	}
    for(unsigned int j=0; j<renderScene->materials.size(); j++){
      materials[j] = renderScene->materials[j];
    }

	/////////////////////////////////////
	////////// Mesh Loading /////////////
	/////////////////////////////////////
	int number_of_faces;
	triangle* tri_faces;
	if (renderScene->meshes.size() >0)
	{
		vbo = renderScene->meshes[0].getVBO();
		vbosize = renderScene->meshes[0].getVBOsize();

		nbo = renderScene->meshes[0].getNBO();
		nbosize= renderScene->meshes[0].getNBOsize();	

		float newcbo[] = {0.0, 1.0, 0.0, 
						0.0, 0.0, 1.0, 
						1.0, 0.0, 0.0};
		cbo = newcbo;
		cbosize = 9;

		ibo = renderScene->meshes[0].getIBO();
		ibosize = renderScene->meshes[0].getIBOsize();

		vector<vector<int>>* temp_faces= renderScene->meshes[0].getFaces();
		//hack
		number_of_faces=ibosize/3;
		tri_faces= new triangle[number_of_faces];

		/*for(int i=0;i<108;i++)
		{
			cout<<vbo[i]<<"   \n";
			if((i+1)%3==0)
			{
				cout<<"\n";
			}
		}*/

		for( int i=0 ; i <number_of_faces ; i++)
		{
			// here P0 has the vertex index of 1 vertex of triangle
			tri_faces[i].p0=glm::vec3(vbo[i*9],vbo[i*9 +1 ],vbo[i*9 +2]);
			tri_faces[i].p1=glm::vec3(vbo[i*9 +3],vbo[i*9 +4],vbo[i*9 +5]);
			tri_faces[i].p2=glm::vec3(vbo[i*9 + 6],vbo[i*9 + 7],vbo[i*9 + 8]);


			tri_faces[i].n0=glm::vec3(nbo[i*9],    nbo[i*9 +1 ],nbo[i*9 +2]);
			tri_faces[i].n1=glm::vec3(nbo[i*9 +3], nbo[i*9 +4], nbo[i*9 +5]);
			tri_faces[i].n2=glm::vec3(nbo[i*9 + 6],nbo[i*9 + 7],nbo[i*9 + 8]);


			////// NOTE This line is hacky, just to save the normal

			tri_faces[i].n0=glm::normalize(tri_faces[i].n0+tri_faces[i].n1+tri_faces[i].n2);


			//tri_faces[i].p0 = glm::vec3(vbo[3*temp_faces[0][i][0]],vbo[3*temp_faces[0][i][0] + 1],vbo[3*(temp_faces[0][i][0]) + 2]);
			//tri_faces[i].p1 = glm::vec3(vbo[3*temp_faces[0][i][1]],vbo[3*temp_faces[0][i][1] + 1],vbo[3*(temp_faces[0][i][1]) + 2]);
			//tri_faces[i].p2 = glm::vec3(vbo[3*temp_faces[0][i][2]],vbo[3*temp_faces[0][i][2] + 1],vbo[3*(temp_faces[0][i][2]) + 2]);
			//cout<"ok";
			//tri_faces[i].p0= vbo[i*tri_faces[i].p0.x, i*tri_faces[i].p0.y,i*tri_faces[i].p0.z];
		}
	
		//vbo[temp_faces[0][0][0]
		/*for( int i=0 ; i <number_of_faces ; i++)
		{
		cout<<tri_faces[i].p0.x<<"	\n";
		cout<<tri_faces[i].p1.x<<"	\n";
		cout<<tri_faces[i].p2.x<<"	\n";
		}*/
	}

	//cout<<renderCam->fov.x<<"fov"<<endl;
	
	// you dont have to do this everytime. think about it
	
	float xDirection = sin(renderCam->yaw) * cos(renderCam->pitch);
	float yDirection = sin(renderCam->pitch);
	float zDirection = cos(renderCam->yaw) * cos(renderCam->pitch);
	glm::vec3 directionToCamera = glm::vec3(xDirection, yDirection, zDirection);
	glm::vec3 viewDirection = -directionToCamera;
	glm::vec3 eyePosition = renderCam->centerPosition + directionToCamera * renderCam->radius;
	renderCam->positions[0]= glm::vec3(eyePosition.x,eyePosition.y,eyePosition.z);
	
	
	// execute the kernel
    cudaRaytraceCore(dptr, renderCam, targetFrame, iterations, materials,
		renderScene->materials.size(), geoms, renderScene->objects.size(),
		vbo,nbo,cbo,vbosize,nbosize,cbosize,objs, renderScene->meshes.size(), number_of_faces, tri_faces,
		ibo,ibosize);
    
    // unmap buffer object
	
    cudaGLUnmapBufferObject(pbo);
  }
else{

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
	  //deleteImage(image);
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
		switch (key) 
		{
		   case(27):
			   exit(1);
			   break;
		   case(' '):
				   cout<<"hello";
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
