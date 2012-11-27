// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <iostream>
#include "scene.h"
#include "objloader.h"

scene::scene(string filename){
	cout << "Reading scene from " << filename << "..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if(fp_in.is_open()){
		while(fp_in.good()){
			string line;
			getline(fp_in, line);
			if(!line.empty()){
				vector<string> tokens = utilityCore::tokenizeString(line);
				if(strcmp(tokens[0].c_str(), "MATERIAL")==0){
					loadMaterial(tokens[1]);
					cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "OBJECT")==0){
					loadObject(tokens[1]);
					cout << " " << endl;
				}else if(strcmp(tokens[0].c_str(), "CAMERA")==0){
					loadCamera();
					cout << " " << endl;
				}
			}
		}
	}
}

int scene::loadObject(string objectid){
	int id = atoi(objectid.c_str());
	if(id!=objects.size()){
		cout << "ERROR: OBJECT ID does not match expected number of objects" << endl;
		return -1;
	}else{
		cout << "Loading Object " << id << "..." << endl;
		geom newObject;
		string line;

		//load object type 
		getline(fp_in,line);
		if (!line.empty() && fp_in.good()){
			if(strcmp(line.c_str(), "sphere")==0){
				cout << "Creating new sphere..." << endl;
				newObject.type = SPHERE;
				newObject.numOfTriangles = 0;
			}else if(strcmp(line.c_str(), "cube")==0){
				cout << "Creating new cube..." << endl;
				newObject.type = CUBE;
				newObject.numOfTriangles = 0;
			}else{
				string objline = line;
				string name;
				string extension;
				istringstream liness(objline);
				getline(liness, name, '.');
				getline(liness, extension, '.');
				if(strcmp(extension.c_str(), "obj")==0){
					bool loadedScene = false;
					cout << "Creating new mesh..." << endl;
					cout << "Reading mesh from " << line << "... " << endl;
					newObject.type = MESH;
					////////////////////////////PARSING OBJ FILE////////////////////////
					vector<glm::vec3> Vertices;
					vector<glm::vec3> Normals;
					vector<vector<string>> Faces;
					vector<vector<vector<string>>> TempFaces;
					vector<vector<vector<int>>> FaceVector;
					////////////////////////////////////////////////////////////////NEW STUFF TO INTEGRATE OBJ LOADER/////////////////////////////////////
					obj* mesh;
					mesh = new obj();
			  		objLoader* loader = new objLoader(line, mesh);
					mesh->buildVBOs();
					delete loader;
				  	loadedScene = true;

				  	if(loadedScene)
				  	{
				  		float* vbo;
				  		int vboSize;
				  		
				  		float* nbo;
				  		int nboSize;
				  		
				  		int* ibo;
				  		int iboSize;

				  		vbo = mesh->getVBO();
						vboSize = mesh->getVBOsize();

						nbo = mesh->getNBO();
						nboSize = mesh->getNBOsize();

						ibo = mesh->getIBO();
						iboSize = mesh->getIBOsize();

				  		int primitivesCount = iboSize/3;
				  		newObject.triangles = new TriangleStruct[primitivesCount];
				  		
				  		for(int index = 0; index < primitivesCount; index++)
				  		{
				  			int iboIndex = 3 * index;
							//printf("\n\n------Primitive Assembly-------");
							TriangleStruct tri;
							tri.index = index;
							
							//Copy Normals
							tri.normal[0] = glm::vec3(nbo[3 * iboIndex], nbo[3 * iboIndex + 1], nbo[3 * iboIndex + 2]);
							tri.normal[1] = glm::vec3(nbo[3 * (iboIndex + 1)], nbo[3 * (iboIndex + 1) + 1], nbo[3 * (iboIndex + 1) + 2]);
							tri.normal[2] = glm::vec3(nbo[3 * (iboIndex + 2)], nbo[3 * (iboIndex + 2) + 1], nbo[3 * (iboIndex + 2) + 2]);
							
							//Copy Vertices
							tri.vertices[0] = glm::vec3(vbo[3 * iboIndex], vbo[3 * iboIndex + 1], vbo[3 * iboIndex + 2]);
							tri.vertices[1] = glm::vec3(vbo[3 * (iboIndex + 1)], vbo[3 * (iboIndex + 1) + 1], vbo[3 * (iboIndex + 1) + 2]);
							tri.vertices[2] = glm::vec3(vbo[3 * (iboIndex + 2)], vbo[3 * (iboIndex + 2) + 1], vbo[3 * (iboIndex + 2) + 2]);

							newObject.triangles[index] = tri;
						}
				  	
					//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

						std::cout << "No. of Vertices = " << vboSize << std::endl;
						std::cout << "No. of Faces = " << primitivesCount << std::endl;
						newObject.numOfTriangles = primitivesCount;
						vbo = NULL;
						nbo = NULL;
						ibo = NULL;
					}

					/////////////////////////////////Parsing Ends Here///////////////////////////////////////
				}else{
					cout << "ERROR: " << line << " is not a valid object type!" << endl;
					return -1;
				}
			}
		}

	//link material
	getline(fp_in,line);
	if(!line.empty() && fp_in.good()){
		vector<string> tokens = utilityCore::tokenizeString(line);
		newObject.materialid = atoi(tokens[1].c_str());
		cout << "Connecting Object " << objectid << " to Material " << newObject.materialid << "..." << endl;
		}

	//load frames
		int frameCount = 0;
	getline(fp_in,line);
	vector<glm::vec3> translations;
	vector<glm::vec3> scales;
	vector<glm::vec3> rotations;
		while (!line.empty() && fp_in.good()){

		//check frame number
		vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
				cout << "ERROR: Incorrect frame count!" << endl;
				return -1;
			}

		//load tranformations
		for(int i=0; i<3; i++){
		glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
		getline(fp_in,line);
		tokens = utilityCore::tokenizeString(line);
		if(strcmp(tokens[0].c_str(), "TRANS")==0){
			translations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
		}else if(strcmp(tokens[0].c_str(), "ROTAT")==0){
			rotations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
		}else if(strcmp(tokens[0].c_str(), "SCALE")==0){
			scales.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
		}
		}

		frameCount++;
		getline(fp_in,line); 
	}

	//move frames into CUDA readable arrays
	newObject.translations = new glm::vec3[frameCount];
	newObject.rotations = new glm::vec3[frameCount];
	newObject.scales = new glm::vec3[frameCount];
	newObject.transforms = new cudaMat4[frameCount];
	newObject.inverseTransforms = new cudaMat4[frameCount];
	for(int i=0; i<frameCount; i++){
		newObject.translations[i] = translations[i];
		newObject.rotations[i] = rotations[i];
		newObject.scales[i] = scales[i];
		glm::mat4 transform = utilityCore::buildTransformationMatrix(translations[i], rotations[i], scales[i]);
		newObject.transforms[i] = utilityCore::glmMat4ToCudaMat4(transform);
		newObject.inverseTransforms[i] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
	}

	//Calculating Bounding Sphere
	if(newObject.type == MESH)
	{
		geom meshObj = newObject;
		glm::vec3 maxBox = glm::vec3(abs(meshObj.triangles[0].vertices[0].x), abs(meshObj.triangles[0].vertices[0].y), abs(meshObj.triangles[0].vertices[0].z));
		glm::vec3 minBox = glm::vec3(-abs(meshObj.triangles[0].vertices[0].x), -abs(meshObj.triangles[0].vertices[0].y), -abs(meshObj.triangles[0].vertices[0].z));
		//Axis Aligned Bounding Sphere (Not scaled equally on all axis)
		for(int i = 0; i < meshObj.numOfTriangles; i++)
		{
			for(int j = 0; j < 3; j++)
			{
				if(meshObj.triangles[i].vertices[j].x > maxBox.x)
					maxBox.x = meshObj.triangles[i].vertices[j].x + 0.1;

				if(meshObj.triangles[i].vertices[j].y > maxBox.y)
					maxBox.y = meshObj.triangles[i].vertices[j].y + 0.1;

				if(meshObj.triangles[i].vertices[j].z > maxBox.z)
					maxBox.z = meshObj.triangles[i].vertices[j].z + 0.1;

				if(meshObj.triangles[i].vertices[j].x < minBox.x)
					minBox.x = meshObj.triangles[i].vertices[j].x - 0.1;

				if(meshObj.triangles[i].vertices[j].y < minBox.y)
					minBox.y = meshObj.triangles[i].vertices[j].y - 0.1;

				if(meshObj.triangles[i].vertices[j].z < minBox.z)
					minBox.z = meshObj.triangles[i].vertices[j].z - 0.1;
			}
		}
		staticGeom BoundingBall;
		BoundingBall.type = SPHERE;
		BoundingBall.rotation = glm::vec3(0.0, 0.0, 0.0);
		meshObj.BSrotation = BoundingBall.rotation;
		BoundingBall.translation = (maxBox + minBox) * 0.5f;
		meshObj.BStranslation = BoundingBall.translation;
		if(abs(maxBox.x - BoundingBall.translation.x) > abs(minBox.x - BoundingBall.translation.x))
			BoundingBall.scale.x = 2.0 * abs(maxBox.x - BoundingBall.translation.x);
		else
			BoundingBall.scale.x = 2.0 * abs(minBox.x - BoundingBall.translation.x);

		if(abs(maxBox.y - BoundingBall.translation.y) > abs(minBox.y - BoundingBall.translation.y))
			BoundingBall.scale.y = 2.0 * abs(maxBox.y - BoundingBall.translation.y);
		else
			BoundingBall.scale.y = 2.0 * abs(minBox.y - BoundingBall.translation.y);

		if(abs(maxBox.z - BoundingBall.translation.z) > abs(minBox.z - BoundingBall.translation.z))
			BoundingBall.scale.z = 2.0 * abs(maxBox.z - BoundingBall.translation.z);
		else
			BoundingBall.scale.z = 2.0 * abs(minBox.z - BoundingBall.translation.z);

		meshObj.BSscale = BoundingBall.scale;
		glm::mat4 transform = utilityCore::buildTransformationMatrix(BoundingBall.translation, BoundingBall.rotation, BoundingBall.scale);
		transform = utilityCore::cudaMat4ToGlmMat4(meshObj.transforms[0]) * transform;
		BoundingBall.transform = utilityCore::glmMat4ToCudaMat4(transform);
		BoundingBall.inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
		meshObj.BSTransform = BoundingBall.transform;
		meshObj.BSInverseTransform = BoundingBall.inverseTransform;

		newObject.BSrotation = meshObj.BSrotation;
		newObject.BStranslation = meshObj.BStranslation;
		newObject.BSscale = meshObj.BSscale;
		newObject.BSTransform = meshObj.BSTransform;
		newObject.BSInverseTransform = meshObj.BSInverseTransform;
	//}

	}
	else
	{
		newObject.BSrotation = glm::vec3(0,0,0);
		newObject.BStranslation = glm::vec3(0,0,0);
		newObject.BSscale = glm::vec3(1,1,1);
		newObject.BSTransform = utilityCore::glmMat4ToCudaMat4(utilityCore::buildTransformationMatrix(newObject.BStranslation, newObject.BSrotation, newObject.BSscale));
		newObject.BSInverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(utilityCore::cudaMat4ToGlmMat4(newObject.BSTransform)));
	}
		objects.push_back(newObject);

	cout << "Loaded " << frameCount << " frames for Object " << objectid << "!" << endl;
		return 1;
	}
}

int scene::loadCamera(){
	cout << "Loading Camera ..." << endl;
		camera newCamera;
	float fovy;

	//load static properties
	for(int i=0; i<4; i++){
		string line;
		getline(fp_in,line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if(strcmp(tokens[0].c_str(), "RES")==0){
			newCamera.resolution = glm::vec2(atoi(tokens[1].c_str()), atoi(tokens[2].c_str()));
		}else if(strcmp(tokens[0].c_str(), "FOVY")==0){
			fovy = atof(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "ITERATIONS")==0){
			newCamera.iterations = atoi(tokens[1].c_str());
		}else if(strcmp(tokens[0].c_str(), "FILE")==0){
			newCamera.imageName = tokens[1];
		}
	}

	//load time variable properties (frames)
		int frameCount = 0;
	string line;
	getline(fp_in,line);
	vector<glm::vec3> positions;
	vector<glm::vec3> views;
	vector<glm::vec3> ups;
		while (!line.empty() && fp_in.good()){

		//check frame number
		vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "frame")!=0 || atoi(tokens[1].c_str())!=frameCount){
				cout << "ERROR: Incorrect frame count!" << endl;
				return -1;
			}

		//load camera properties
		for(int i=0; i<3; i++){
		//glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
		getline(fp_in,line);
		tokens = utilityCore::tokenizeString(line);
		if(strcmp(tokens[0].c_str(), "EYE")==0){
			positions.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
		}else if(strcmp(tokens[0].c_str(), "VIEW")==0){
			views.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
		}else if(strcmp(tokens[0].c_str(), "UP")==0){
			ups.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
		}
		}

		frameCount++;
		getline(fp_in,line); 
	}
	newCamera.frames = frameCount;

	//move frames into CUDA readable arrays
	newCamera.positions = new glm::vec3[frameCount];
	newCamera.views = new glm::vec3[frameCount];
	newCamera.ups = new glm::vec3[frameCount];
	for(int i=0; i<frameCount; i++){
		newCamera.positions[i] = positions[i];
		newCamera.views[i] = glm::normalize(views[i]);
		newCamera.ups[i] = glm::normalize(ups[i]);
	}

	//calculate fov based on resolution
	float yscaled = tan(fovy*(PI/180));
	float xscaled = (yscaled * newCamera.resolution.x)/newCamera.resolution.y;
	float fovx = (atan(xscaled)*180)/PI;
	newCamera.fov = glm::vec2(fovx, fovy);

	renderCam = newCamera;

	//set up render camera stuff
	renderCam.image = new glm::vec3[(int)renderCam.resolution.x*(int)renderCam.resolution.y];
	renderCam.rayList = new ray[(int)renderCam.resolution.x*(int)renderCam.resolution.y];
	for(int i=0; i<renderCam.resolution.x*renderCam.resolution.y; i++){
		renderCam.image[i] = glm::vec3(0,0,0);
	}

	cout << "Loaded " << frameCount << " frames for camera!" << endl;
	return 1;
}

int scene::loadMaterial(string materialid){
	int id = atoi(materialid.c_str());
	if(id!=materials.size()){
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}else{
		cout << "Loading Material " << id << "..." << endl;
		material newMaterial;

		//load static properties
		for(int i=0; i<10; i++){
			string line;
			getline(fp_in,line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if(strcmp(tokens[0].c_str(), "RGB")==0){
				glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.color = color;
			}else if(strcmp(tokens[0].c_str(), "SPECEX")==0){
				newMaterial.specularExponent = atof(tokens[1].c_str());				  
			}else if(strcmp(tokens[0].c_str(), "SPECRGB")==0){
				glm::vec3 specColor( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.specularColor = specColor;
			}else if(strcmp(tokens[0].c_str(), "REFL")==0){
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "REFR")==0){
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "REFRIOR")==0){
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());					  
			}else if(strcmp(tokens[0].c_str(), "SCATTER")==0){
				newMaterial.hasScatter = atof(tokens[1].c_str());
			}else if(strcmp(tokens[0].c_str(), "ABSCOEFF")==0){
				glm::vec3 abscoeff( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
				newMaterial.absorptionCoefficient = abscoeff;
			}else if(strcmp(tokens[0].c_str(), "RSCTCOEFF")==0){
				newMaterial.reducedScatterCoefficient = atof(tokens[1].c_str());					  
			}else if(strcmp(tokens[0].c_str(), "EMITTANCE")==0){
				newMaterial.emittance = atof(tokens[1].c_str());					  

			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}