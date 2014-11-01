#include <nau.h>
#include <nau/config.h>

// added for directory loading
#ifdef NAU_PLATFORM_WIN32
#include <nau/system/dirent.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif
#include <nau/math/vec4.h>

#include <ctime>

#include <GL/glew.h>

#include <nau/system/file.h>
#include <nau/scene/scenefactory.h>
#include <nau/loader/cboloader.h>
#include <nau/loader/objLoader.h>
#include <nau/loader/ogremeshloader.h>
#include <nau/loader/assimploader.h>
#include <nau/loader/patchLoader.h>
#include <nau/loader/projectloader.h>
#include <nau/world/worldfactory.h>
#include <nau/event/eventFactory.h>
#include <nau/resource/fontmanager.h>
#include "nau/slogger.h"

#include <nau/debug/profile.h>
#include <nau/debug/state.h>
//#include <nau/debug/fonts.h>


#ifdef GLINTERCEPTDEBUG
#include <nau/loader/projectloaderdebuglinker.h>
#endif //GLINTERCEPTDEBUG

using namespace nau;
using namespace nau::system;
using namespace nau::loader;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::resource;
using namespace nau::world;
using namespace nau::material;

/*
 * Improvements:
 * - Move viewports into rendermanager
 */

static nau::Nau *gInstance = 0;

nau::Nau*
Nau::create (void) {
	if (0 != gInstance) {
		throw (NauInstanciationError ("An instance of Nau already exists"));
	}
	gInstance = new Nau;

	return gInstance;
}

nau::Nau*
Nau::getInstance (void) {
	if (0 == gInstance) {
		create();
		//throw (NauInstanciationError ("No instance of Nau exists"));
	}

	return gInstance;
}

Nau::Nau() :
	m_WindowWidth (0.0f), 
	m_WindowHeight (0.0f), 
	m_vViewports(),
	m_Inited (false),
	m_Physics (false),
	loadedScenes(0),
	m_ActiveCameraName(""),
	m_Name("Nau"),
	m_RenderFlags(COUNT_RENDER_FLAGS),
	m_UseTangents(false),
	m_UseTriangleIDs(false),
	m_CoreProfile(false),
	isFrameBegin(true)
{
	// create a default black viewport
//	createViewport("default");
}

Nau::~Nau()
{
	clear();

}

bool 
Nau::init (bool context, std::string aConfigFile)
{
	bool result;
	if (true == context) {

		m_pRenderManager = new RenderManager;
		result = m_pRenderManager->init();
		if (!result)
			return(0);

		m_pEventManager = new EventManager;
	}	
	
	m_pResourceManager = new ResourceManager ("."); /***MARK***/ //Get path!!!
	m_pMaterialLibManager = new MaterialLibManager();

	try {
		ProjectLoader::loadMatLib("./nauSystem.mlib");
	}
	catch (std::string s) {
		clear();
		throw(s);
	}

	FontManager::addFont("CourierNew10", "./couriernew10.xml", "__FontCourierNew10");

//	m_ProfileMaterial =	MATERIALLIBMANAGER->getDefaultMaterial("__Emission White");


	//m_pScene = SceneFactory::create ("Octree"); /***MARK***/ //Check for 0. Configuration
	m_pWorld = WorldFactory::create ("Bullet");

//	m_pWorld->setScene (m_pScene);
	
//	m_pConsole = m_pRenderManager->createConsole (createViewport());

	CLOCKS_PER_MILISEC = CLOCKS_PER_SEC / 1000.0;
	INV_CLOCKS_PER_MILISEC = 1.0 / CLOCKS_PER_MILISEC;

	m_CurrentTime = clock() * INV_CLOCKS_PER_MILISEC;
	m_LastFrameTime = NO_TIME;

//	Profile::init();

	m_Inited = true;

	EVENTMANAGER->addListener("WINDOW_SIZE_CHANGED",this);

	m_DefaultState = IState::create();
	m_Viewport = createViewport("defaultFixedVP");

	//Init state reader's function list
	State::init();

	return true;
}

std::string&
Nau::getName() 
{
	return(m_Name);
}

//
//		USER ATTRIBUTES
//

bool 
Nau::validateUserAttribContext(std::string context) {

	if (context == "LIGHT" || context == "CAMERA" || context == "VIEWPORT"
		|| context == "TEXTURE" || context == "STATE" || context == "VIEWPORT" || context == "PASS")
		return true;

	return false;
}


AttribSet *
Nau::getAttribs(std::string context) {

	AttribSet *attribs = NULL;

	if (context == "LIGHT")
		attribs = &(Light::Attribs);
	else if (context == "CAMERA")
		attribs = &(Camera::Attribs);
	else if (context == "VIEWPORT")
		attribs = &(Viewport::Attribs);
	else if (context == "TEXTURE")
		attribs = &(Texture::Attribs);
	else if (context == "STATE")
		attribs = &(IState::Attribs);
	else if (context == "VIEWPORT")
		attribs = &(Viewport::Attribs);
	else if (context == "PASS")
		attribs = &(Pass::Attribs);

	return attribs;
}


bool 
Nau::validateUserAttribName(std::string context, std::string name) {

	AttribSet *attribs = getAttribs(context);

 // invalid context
	if (attribs == NULL)
		return false;

	Attribute a = attribs->get(name);
	if (a.getName() == "NO_ATTR")
		return true;
	else
		return false;
}


//
//		EVENTS
//


void
Nau::eventReceived(const std::string &sender, const std::string &eventType, IEventData *evtData) 
{
	if (eventType == "WINDOW_SIZE_CHANGED") {
	
		vec3 *evVec = (vec3 *)evtData->getData();
		setWindowSize(evVec->x,evVec->y);
	}
}

void
Nau::readProjectFile (std::string file, int *width, int *height)
{
	//clear();

	try {
		ProjectLoader::load (file, width, height, &m_UseTangents, &m_UseTriangleIDs);
		isFrameBegin = true;
	}
	catch (std::string s) {
		clear();
		throw(s);
	}

	setActiveCameraName(RENDERMANAGER->getDefaultCameraName());
		
	m_pWorld->setScene(RENDERMANAGER->getScene("Terrain"));

	if (m_UseTriangleIDs)
		RENDERMANAGER->prepareTriangleIDs(true);
}

void
Nau::readDirectory (std::string dirName)
{
	DIR *dir;
	struct dirent *ent;
	bool result = true;
	char fileName [1024];
	char sceneName[256];

	clear();
	sprintf(sceneName,"MainScene"); //,loadedScenes);
	loadedScenes++;
	RENDERMANAGER->createScene (sceneName);
	dir = opendir (dirName.c_str());

	if (0 == dir) {
		NAU_THROW("Can't open dir: %s",dirName);
	}
	while (0 != (ent = readdir (dir))) {

#ifdef NAU_PLATFORM_WIN32
		sprintf (fileName, "%s\\%s", dirName.c_str(), ent->d_name);
#else
		sprintf (fileName, "%s/%s", dirName, ent->d_name);						
#endif
		try {
			NAU->loadAsset (fileName, sceneName);
		}
		catch(std::string &s) {
			closedir(dir);
			throw(s);
		}
	}
	closedir (dir);
	loadFilesAndFoldersAux(sceneName,false);	
	
}

void
Nau::clear() {

	setActiveCameraName("");
	SceneObject::ResetCounter();
	MATERIALLIBMANAGER->clear();
	EVENTMANAGER->clear(); 
	EVENTMANAGER->addListener("WINDOW_SIZE_CHANGED",this);
	RENDERMANAGER->clear();
	RESOURCEMANAGER->clear();
	
	// Need to clear font manager

	while (!m_vViewports.empty()){
	
		m_vViewports.erase(m_vViewports.begin());
	}

	m_Viewport = createViewport("defaultFixedVP");

	//_CrtDumpMemoryLeaks();

	ProjectLoader::loadMatLib("./nauSystem.mlib");
}

void
Nau::readModel (std::string fileName) throw (std::string)
{
	clear();
	bool result = true;
	
	char sceneName[256];

	sprintf(sceneName,"MainScene"); //,loadedScenes);
	loadedScenes++;

	RENDERMANAGER->createScene (sceneName);

	try {
		NAU->loadAsset (fileName, sceneName);
		loadFilesAndFoldersAux(sceneName, false);
	} 
	catch (std::string &s) {
			throw(s);
	}
}


void
Nau::appendModel(std::string fileName)

{
	
	char sceneName[256];

	sprintf(sceneName,"MainScene"); //,loadedScenes);
	loadedScenes++;

	RENDERMANAGER->createScene (sceneName);

	try {
		NAU->loadAsset (fileName, sceneName);
		loadFilesAndFoldersAux(sceneName, false);
	}
	catch( std::string &s){
		throw(s);
	}
}


void Nau::loadFilesAndFoldersAux(char *sceneName, bool unitize) {

	Camera *aNewCam = RENDERMANAGER->getCamera ("MainCamera");
	Viewport *v = NAU->getViewport("defaultFixedVP");//createViewport ("MainViewport", nau::math::vec4(0.0f, 0.0f, 0.0f, 1.0f));
	aNewCam->setViewport (v);

	setActiveCameraName("MainCamera");

	if (unitize) {
		aNewCam->setPerspective (60.0f, 0.01f, 100.0f);
		aNewCam->setProp(Camera::POSITION, 0.0f, 0.0f, 5.0f, 1.0f);
	}
	else
		aNewCam->setPerspective (60.0f, 1.0f, 10000.0f);

	// creates a directional light by default
	Light *l = RENDERMANAGER->getLight ("MainDirectionalLight");
	l->setProp(Light::DIRECTION,1.0f,-1.0f,-1.0f, 0.0f);
	l->setProp(Light::COLOR, 0.9f,0.9f,0.9f,1.0f);
	l->setProp(Light::AMBIENT,0.5f,0.5f,0.5f,1.0f );

	Pipeline *aPipeline = RENDERMANAGER->getPipeline ("MainPipeline");
	Pass *aPass = aPipeline->createPass("MainPass");
	aPass->setCamera ("MainCamera");

	aPass->setViewport (v);
	aPass->setPropb(Pass::COLOR_CLEAR, true);
	aPass->setPropb(Pass::DEPTH_CLEAR, true);
	aPass->addLight ("MainDirectionalLight");

	aPass->addScene(sceneName);
	RENDERMANAGER->setActivePipeline ("MainPipeline");

	//std::vector<std::string> *materialNames = MATERIALLIBMANAGER->getMaterialNames (DEFAULTMATERIALLIBNAME);
	//RENDERMANAGER->materialNamesFromLoadedScenes (*materialNames); 
	//delete materialNames;

//	RENDERMANAGER->prepareTriangleIDsAndTangents(true,true);

}

bool
Nau::reload (void)
{
	if (false == m_Inited) {
		return false;
	}
	//RenderManager->reload();

	return true;
}

//void 
//nau::Nau::setProfileMaterial(std::string aMaterial) {
//
//	m_ProfileMaterial = MATERIALLIBMANAGER->getDefaultMaterial(aMaterial);
//	if (m_ProfileMaterial == 0)
//		m_ProfileMaterial = MATERIALLIBMANAGER->getDefaultMaterial("__Emission White");
//}


void
Nau::step(int count)
{
	IRenderer *renderer = RENDERER;
	m_CurrentTime = clock() * INV_CLOCKS_PER_MILISEC;
	if (NO_TIME == m_LastFrameTime) {
		m_LastFrameTime = m_CurrentTime;
	}
	double deltaT = m_CurrentTime - m_LastFrameTime;
	m_LastFrameTime = m_CurrentTime;
	unsigned char pipeEventFlag;

	renderer->resetCounters();

	for (int i = 0; i < count || count <= 0; i++){

		if (isFrameBegin){
			m_pEventManager->notifyEvent("FRAME_BEGIN", "Nau", "", NULL);
#ifdef GLINTERCEPTDEBUG
			addMessageToGLILog("\n#NAU(FRAME,START)");
#endif //GLINTERCEPTDEBUG
			isFrameBegin = false;
		}


		//if (m_Animations.size() > 0) {
		//	std::map<std::string, nau::animation::IAnimation*>::iterator animIter;
		//	
		//	animIter = m_Animations.begin();

		//	for (; animIter != m_Animations.end(); animIter++) {
		//		if (false == (*animIter).second->isFinished()) {
		//			(*animIter).second->step (static_cast<float> (deltaT));
		//		} else {
		//			delete ((*animIter).second);
		//			m_Animations.erase (animIter);
		//		}
		//	}
		//}

		//### Será que deve ser por pass?
		if (true == m_Physics) {
			m_pWorld->update();
			m_pEventManager->notifyEvent("DYNAMIC_CAMERA", "MainCanvas", "", NULL);
		}

		//renderer->setDefaultState();

#ifdef GLINTERCEPTDEBUG
		Pass *renderPass = NULL;
		if (m_pRenderManager->getNumPipelines() > 0){
			renderPass = m_pRenderManager->getCurrentPass();
		}
		if (renderPass){
			addMessageToGLILog(("\n#NAU(PASS,START," + renderPass->getName() + ")").c_str());
		}
#endif //GLINTERCEPTDEBUG
		pipeEventFlag = m_pRenderManager->renderActivePipeline();
#ifdef GLINTERCEPTDEBUG
		if (renderPass){
			addMessageToGLILog(("\n#NAU(PASS,END," + renderPass->getName() + ")").c_str());
		}
#endif //GLINTERCEPTDEBUG

#ifdef NAU_RENDER_FLAGS
		//#ifdef PROFILE
		//	if (getRenderFlag(Nau::PROFILE_RENDER_FLAG))
		//	{
		//		PROFILE ("Profile rendering");
		//
		//		renderer->setViewport(m_WindowWidth, m_WindowHeight);
		//
		//		renderer->saveAttrib(IRenderer::RENDER_MODE);
		//		renderer->setRenderMode(IRenderer::MATERIAL_MODE);
		//		
		//		m_ProfileMaterial->prepare();
		//		renderer->disableDepthTest();
		//		//RENDERER->enableTexturing();
		//		setOrthographicProjection (static_cast<int>(m_WindowWidth), 
		//										static_cast<int>(m_WindowHeight));
		//
		//		Profile::dumpLevelsOGL();
		//
		// 		char s[128];
		// 		sprintf (s, "Primitives: %d", RENDERER->getTriCount());
		// 		renderBitmapString (30,400, Profile::font,s);
		//		
		//		resetPerspectiveProjection();
		//		renderer->enableDepthTest();
		//		renderer->restoreAttrib();
		//	}
		//#endif // PROFILE
#endif // NAU_RENDER_FLAGS
		switch (pipeEventFlag){
		case PIPE_PASS_STARTEND:
		case PIPE_PASS_END:
			m_pEventManager->notifyEvent("FRAME_END", "Nau", "", NULL);
#ifdef GLINTERCEPTDEBUG
			addMessageToGLILog("\n#NAU(FRAME,END)");
#endif //GLINTERCEPTDEBUG
			isFrameBegin = true;
			break;
		}

		if (count <= 0){
			break;
		}
	}
}


void 
Nau::setActiveCameraName(const std::string &aCamName) {

	if (m_ActiveCameraName != "") {
		EVENTMANAGER->removeListener("CAMERA_MOTION",RENDERMANAGER->getCamera(m_ActiveCameraName));
		EVENTMANAGER->removeListener("CAMERA_ORIENTATION",RENDERMANAGER->getCamera(m_ActiveCameraName));
	}
	m_ActiveCameraName = aCamName;

	if (m_ActiveCameraName != "") {
		EVENTMANAGER->addListener("CAMERA_MOTION",RENDERMANAGER->getCamera(m_ActiveCameraName));
		EVENTMANAGER->addListener("CAMERA_ORIENTATION",RENDERMANAGER->getCamera(m_ActiveCameraName));
	}
}


nau::scene::Camera *
Nau::getActiveCamera() {

	if (RENDERMANAGER->hasCamera(m_ActiveCameraName)) {
		return RENDERMANAGER->getCamera(m_ActiveCameraName);
	}
	else
		return NULL;
}


float
Nau::getDepthAtCenter() {

	float f;
	vec2 v = m_Viewport->getPropf2(Viewport::ORIGIN);
	v += m_Viewport->getPropf2(Viewport::SIZE);
	Camera *cam = RENDERMANAGER->getCamera(m_ActiveCameraName);
	float a = cam->getPropm4(Camera::PROJECTION_MATRIX).at(2,2);
	float b = cam->getPropm4(Camera::PROJECTION_MATRIX).at(3,2);
	// This must move to glRenderer!!!!
	f = RENDERER->getDepthAtPoint(v.x/2, v.y/2);
	f = (f-0.5) * 2;
	SLOG("Depth %f %f\n",b/(f+a), f); 
	return (0.5 * (-a*f + b) / f + 0.5);
}


void 
Nau::sendKeyToEngine (char keyCode)
{
	switch(keyCode) {
	case 'K':
		Profile::Reset();
		break;
	case 'I':
		getDepthAtCenter();
		break;
	}

	//RenderManager->sendKeyToEngine (keyCode);
}



IWorld&
Nau::getWorld (void)
{
	return (*m_pWorld);
}



void
Nau::loadAsset (std::string aFilename, std::string sceneName, std::string params) throw (std::string)
{
	File file (aFilename);

	try {
		switch (file.getType()) {
			case File::PATCH:
			{
				PatchLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath());
				break;
			}
			case File::COLLADA:
			{
				AssimpLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath(),params);
				//std::string uri (file.getURI());
				//ColladaLoader::loadScene (RENDERMANAGER->getScene (sceneName), uri);
				break;
			}
			case File::NAUBINARYOBJECT:
			{
				std::string filepath (file.getFullPath());
				CBOLoader::loadScene (RENDERMANAGER->getScene (sceneName), filepath);
				break;
			}
			case File::THREEDS:
			{
				AssimpLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath(), params);
				//THREEDSLoader::loadScene (RENDERMANAGER->getScene (sceneName), file.getFullPath(),params);
				
				break;
			}
			case File::WAVEFRONTOBJ:
			{
				AssimpLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath(),params);
				//OBJLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath());
				
				break;
			}
			case File::OGREXMLMESH:
			{
				OgreMeshLoader::loadScene(RENDERMANAGER->getScene (sceneName), file.getFullPath());
				
				break;
			}

			default:
			  break;
		}
	}
	catch(std::string &s) {
		throw(s);
	}

	Profile::Reset();
}

void
Nau::writeAssets (std::string fileType, std::string aFilename, std::string sceneName)
{
	if (0 == fileType.compare ("CBO")) {
		CBOLoader::writeScene (RENDERMANAGER->getScene (sceneName), aFilename);
	}
}

void
Nau::setWindowSize (float width, float height)
{
	m_Viewport->setProp(Viewport::SIZE, vec2(width,height));
	m_WindowWidth = width;
	m_WindowHeight = height;
}

float 
Nau::getWindowHeight() 
{
	return(m_WindowHeight);
}

float 
Nau::getWindowWidth() 
{
	return(m_WindowWidth);
}



Viewport*
Nau::createViewport (const std::string &name, const nau::math::vec4 &bgColor) 
{
	Viewport* v = new Viewport;

	v->setName(name);
	v->setProp (Viewport::ORIGIN, vec2(0.0f,0.0f));
	v->setProp (Viewport::SIZE, vec2(m_WindowWidth, m_WindowHeight));

	v->setProp(Viewport::CLEAR_COLOR, bgColor);

	m_vViewports[name] = v;

	return v;
}

Viewport*
Nau::createViewport (const std::string &name) 
{
	Viewport* v = new Viewport;

	v->setName(name);
	v->setProp (Viewport::ORIGIN, vec2(0.0f,0.0f));
	v->setProp (Viewport::SIZE, vec2(m_WindowWidth, m_WindowHeight));

	m_vViewports[name] = v;

	return v;
}


Viewport* 
Nau::getViewport (const std::string &name)
{
	if (m_vViewports.count(name))
		return m_vViewports[name];
	else
		return NULL;
}



Viewport*
Nau::getDefaultViewport() {
	
	return m_Viewport;
}


std::vector<std::string> *
Nau::getViewportNames()
{
	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::map<std::string, nau::render::Viewport*>::iterator iter = m_vViewports.begin(); iter != m_vViewports.end(); ++iter ) {
      names->push_back(iter->first); 
    }
	return names;
}

void
Nau::enablePhysics (void) 
{
	m_Physics = true;
}

void
Nau::disablePhysics (void)
{
	m_Physics = false;
}


void
Nau::setRenderFlag(RenderFlags aFlag, bool aState) 
{
	m_RenderFlags[aFlag] = aState;
}


bool
Nau::getRenderFlag(RenderFlags aFlag)
{
	return(m_RenderFlags[aFlag]);
}

int 
Nau::picking (int x, int y, std::vector<nau::scene::SceneObject*> &objects, nau::scene::Camera &aCamera)
{
	return -1;//	RenderManager->pick (x, y, objects, aCamera);
}

//StateList functions:
void 
Nau::loadStateXMLFile(std::string file){
	State::loadStateXMLFile(file);
}
std::vector<std::string> 
Nau::getStateEnumNames(){
	return State::getStateEnumNames();
}
std::string 
Nau::getState(std::string enumName){
	return State::getState(enumName);
}

//void 
//Nau::addAnimation (std::string animationName, nau::animation::IAnimation *anAnimation)
//{
//	m_Animations [animationName] = anAnimation; //Attention! Possibility of memory leak
//}
//
//nau::animation::IAnimation*
//Nau::getAnimation (std::string animationName)
//{
//	if (m_Animations.count (animationName) > 0) {
//		return m_Animations [animationName];
//	}
//	
//	return 0;
//}

//void 
//Nau::enableStereo (void)
//{
//	RENDERMANAGER->enableStereo();
//}
//
//void 
//Nau::disableStereo (void)
//{
//	RENDERMANAGER->disableStereo();
//}

RenderManager* 
Nau::getRenderManager (void)
{
	return m_pRenderManager;
}

ResourceManager* 
Nau::getResourceManager (void)
{
	return m_pResourceManager;
}

MaterialLibManager*
Nau::getMaterialLibManager (void)
{
	return m_pMaterialLibManager;
}

EventManager*
Nau::getEventManager (void)
{
	return m_pEventManager;
}


///* =================================================
//
//	DATA TYPES
//
//================================================= */
//
//std::string 
//Nau::getDataType(Enums::DataType dt) 
//{
//	switch(dt) {
//		case Enums::DataType::INT: return("INT");
//		case SAMPLER: return("SAMPLER");
//		case BOOL: return("BOOL");
//		case IVEC2: return("IVEC2");
//		case IVEC3: return("IVEC3");
//		case IVEC4: return("IVEC4");
//		case BVEC2: return("BVEC2");
//		case BVEC3: return("BVEC3");
//		case BVEC4: return("BVEC4");
//		case FLOAT: return("FLOAT");
//		case VEC2: return("VEC2");
//		case VEC3: return("VEC3");
//		case VEC4: return("VEC4");
//		case MAT3: return("MAT3");
//		case MAT4: return("MAT4");
//		default: return ("FLOAT");
//	}
//}
//
//Nau::DataType 
//Nau::getDataType(std::string &s)
//{
//	if ("INT" == s)
//		return INT;
//	else if ("SAMPLER" == s)
//		return SAMPLER;
//	else if ("BOOL" == s)
//		return BOOL;
//	else if ("IVEC2" == s)
//		return IVEC2;
//	else if ("IVEC3" == s)
//		return IVEC3;
//	else if ("IVEC4" == s)
//		return IVEC4;
//	else if ("BVEC2" == s)
//		return BVEC2;
//	else if ("BVEC3" == s)
//		return BVEC3;
//	else if ("BVEC4" == s)
//		return BVEC4;
//	else if ("FLOAT" == s)
//		return FLOAT;
//	else if ("VEC2" == s)
//		return VEC2;
//	else if ("VEC3" == s)
//		return VEC3;
//	else if ("VEC4" == s)
//		return VEC4;
//	else if ("MAT3" == s)
//		return MAT3;
//	else if ("MAT4" == s)
//		return MAT4;
//	else
//		return FLOAT;
//}
