#include <sstream>

#include <nau/render/pass.h>
#include <nau/geometry/axis.h>
#include <nau/geometry/frustum.h>
#include <nau/debug/profile.h>
#include <nau.h>

using namespace nau::material;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::geometry;



AttribSet Pass::Attribs;
bool Pass::Inited = Pass::Init();

bool
Pass::Init() {

	// BOOL
	Attribs.add(Attribute(COLOR_CLEAR, "COLOR_CLEAR", Enums::DataType::BOOL, false, new bool(true)));
	Attribs.add(Attribute(COLOR_ENABLE, "COLOR_ENABLE", Enums::DataType::BOOL, false, new bool(true)));
	Attribs.add(Attribute(DEPTH_CLEAR, "DEPTH_CLEAR", Enums::DataType::BOOL, false, new bool(true)));
	Attribs.add(Attribute(DEPTH_ENABLE, "DEPTH_ENABLE", Enums::DataType::BOOL, false, new bool(true)));
	Attribs.add(Attribute(DEPTH_MASK, "DEPTH_MASK", Enums::DataType::BOOL, false, new bool(true)));
	Attribs.add(Attribute(DEPTH_CLAMPING, "DEPTH_CLAMPING", Enums::DataType::BOOL, false, new bool(false)));
	Attribs.add(Attribute(STENCIL_CLEAR, "STENCIL_CLEAR", Enums::DataType::BOOL, false, new bool(true)));
	Attribs.add(Attribute(STENCIL_ENABLE, "STENCIL_ENABLE", Enums::DataType::BOOL, false, new bool(false)));

	// ENUM
	Attribs.add(Attribute(RUN_MODE, "RUN_MODE", Enums::DataType::ENUM, true, new int(RUN_ALWAYS)));
	Attribs.listAdd("RUN_MODE", "DONT_RUN", DONT_RUN);
	Attribs.listAdd("RUN_MODE", "RUN_ALWAYS", RUN_ALWAYS);
	Attribs.listAdd("RUN_MODE", "SKIP_FIRST_FRAME", SKIP_FIRST_FRAME);
	Attribs.listAdd("RUN_MODE", "RUN_ONCE", RUN_ONCE);

	Attribs.add(Attribute(STENCIL_FUNC, "STENCIL_FUNC", Enums::DataType::ENUM, false, new int(ALWAYS)));
	Attribs.listAdd("STENCIL_FUNC", "LESS", LESS);
	Attribs.listAdd("STENCIL_FUNC", "NEVER", NEVER);
	Attribs.listAdd("STENCIL_FUNC", "ALWAYS", ALWAYS);
	Attribs.listAdd("STENCIL_FUNC", "LEQUAL", LEQUAL);
	Attribs.listAdd("STENCIL_FUNC", "EQUAL", EQUAL);
	Attribs.listAdd("STENCIL_FUNC", "GEQUAL", GEQUAL);
	Attribs.listAdd("STENCIL_FUNC", "GREATER", GREATER);
	Attribs.listAdd("STENCIL_FUNC", "NOT_EQUAL", NOT_EQUAL);

	Attribs.add(Attribute(STENCIL_FAIL, "STENCIL_FAIL", Enums::DataType::ENUM, false, new int(KEEP)));
	Attribs.listAdd("STENCIL_FAIL", "KEEP", KEEP);
	Attribs.listAdd("STENCIL_FAIL", "ZERO", ZERO);
	Attribs.listAdd("STENCIL_FAIL", "REPLACE", REPLACE);
	Attribs.listAdd("STENCIL_FAIL", "INCR", INCR);
	Attribs.listAdd("STENCIL_FAIL", "INCR_WRAP", INCR_WRAP);
	Attribs.listAdd("STENCIL_FAIL", "DECR", DECR);
	Attribs.listAdd("STENCIL_FAIL", "DECR_WRAP", DECR_WRAP);
	Attribs.listAdd("STENCIL_FAIL", "INVERT", INVERT);

	Attribs.add(Attribute(STENCIL_DEPTH_FAIL, "STENCIL_DEPTH_FAIL", Enums::DataType::ENUM, false, new int(KEEP)));
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "KEEP", KEEP);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "ZERO", ZERO);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "REPLACE", REPLACE);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "INCR", INCR);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "INCR_WRAP", INCR_WRAP);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "DECR", DECR);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "DECR_WRAP", DECR_WRAP);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "INVERT", INVERT);

	Attribs.add(Attribute(STENCIL_DEPTH_PASS, "STENCIL_DEPTH_PASS", Enums::DataType::ENUM, false, new int(KEEP)));
	Attribs.listAdd("STENCIL_DEPTH_PASS", "KEEP", KEEP);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "ZERO", ZERO);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "REPLACE", REPLACE);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "INCR", INCR);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "INCR_WRAP", INCR_WRAP);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "DECR", DECR);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "DECR_WRAP", DECR_WRAP);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "INVERT", INVERT);

	Attribs.add(Attribute(DEPTH_FUNC, "DEPTH_FUNC", Enums::DataType::ENUM, false, new int(LESS)));
	Attribs.listAdd("DEPTH_FUNC", "LESS", LESS);
	Attribs.listAdd("DEPTH_FUNC", "NEVER", NEVER);
	Attribs.listAdd("DEPTH_FUNC", "ALWAYS", ALWAYS);
	Attribs.listAdd("DEPTH_FUNC", "LEQUAL", LEQUAL);
	Attribs.listAdd("DEPTH_FUNC", "EQUAL", EQUAL);
	Attribs.listAdd("DEPTH_FUNC", "GEQUAL", GEQUAL);
	Attribs.listAdd("DEPTH_FUNC", "GREATER", GREATER);
	Attribs.listAdd("DEPTH_FUNC", "NOT_EQUAL", NOT_EQUAL);

	// VEC4
	Attribs.add(Attribute(COLOR_CLEAR_VALUE, "COLOR_CLEAR_VALUE", Enums::DataType::VEC4, false, new vec4()));

	// FLOAT
	Attribs.add(Attribute(DEPTH_CLEAR_VALUE, "DEPTH_CLEAR_VALUE", Enums::DataType::FLOAT, false, new float(1.0f)));
	Attribs.add(Attribute(STENCIL_CLEAR_VALUE, "STENCIL_CLEAR_VALUE", Enums::DataType::FLOAT, false, new float(0.0f)));

	//INT
	Attribs.add(Attribute(STENCIL_OP_REF, "STENCIL_OP_REF", Enums::DataType::INT, false, new int(0)));

	//UINT
	Attribs.add(Attribute(STENCIL_OP_MASK, "STENCIL_OP_MASK", Enums::DataType::UINT, false, new unsigned int(255)));

	NAU->registerAttributes("PASS", &Attribs);

	return true;
}


Pass::Pass (const std::string &passName) :
	m_ClassName("default"),
	m_Name (passName),
	m_CameraName ("default"),
	m_SceneVector(),
	m_MaterialMap(),
	m_Viewport (0),
	m_RestoreViewport (0),
	m_RemapMode (REMAP_DISABLED)
{
	registerAndInitArrays("PASS", Attribs);

	initVars();
	EVENTMANAGER->addListener("SCENE_CHANGED",this);
}


Pass::~Pass() {

}


void
Pass::eventReceived(const std::string &sender, const std::string &eventType, IEventData *evtData) 
{
	if (eventType == "SCENE_CHANGED") 
		updateMaterialMaps(sender);
}


const std::string &
Pass::getClassName() {

	return m_ClassName;
}


std::string &
Pass::getName (void)
{
	return m_Name;
}



void 
Pass::initVars() {

	m_RenderTarget = NULL;

	m_RTSizeWidth = 512; // size of render targets
	m_RTSizeHeight = 512;

	m_UseRT = false; // enable, disable
}



// --------------------------------------------------
//		RENDER TEST
// --------------------------------------------------

void
Pass::setMode(RunMode value) {

	m_EnumProps[RUN_MODE] = value;
}


bool
Pass::renderTest(void) {

	// most common case: run pass in all frames
	if (m_EnumProps[RUN_MODE] == RUN_ALWAYS)
		return true;

	// pass disabled
	else if (m_EnumProps[RUN_MODE] == DONT_RUN)
		return false;

	else {
		unsigned long f = NAU->getFrameCount();
		bool even = (f % 2 == 0);
		// check for skip_first and run_once cases
		if ((m_EnumProps[RUN_MODE] == SKIP_FIRST_FRAME && (f == 0)) || (m_EnumProps[RUN_MODE] == RUN_ONCE && (f > 0)))
			return false;
		else
			return true;
	}
}


// --------------------------------------------------
//		PREPARE, DO, RESTORE
// --------------------------------------------------


void
Pass::prepareBuffers() {

	RENDERER->prepareBuffers(this);
}


void
Pass::prepare (void) {

	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->bind();
	}

	setupCamera();
	prepareBuffers();
	setupLights();
}




void
Pass::doPass (void) {

	Camera *aCam = 0;
	Frustum camFrustum;
	std::vector<SceneObject*>::iterator objsIter;
	std::vector<std::string>::iterator scenesIter;
	std::vector<nau::scene::SceneObject*> sceneObjects;

	const float *a = RENDERER->getMatrix(IRenderer::PROJECTION_VIEW_MODEL);
	camFrustum.setFromMatrix (a);
	aCam = RENDERMANAGER->getCamera (m_CameraName);
	RENDERMANAGER->clearQueue();

	scenesIter = m_SceneVector.begin();

	for ( ; scenesIter != m_SceneVector.end(); ++scenesIter) {

		IScene *aScene = RENDERMANAGER->getScene (*scenesIter);
		{
			PROFILE("View Frustum Culling");
			sceneObjects = aScene->findVisibleSceneObjects(camFrustum, *aCam);
		//	sceneObjects = aScene->getAllObjects();
		}
		objsIter = sceneObjects.begin();
		for ( ; objsIter != sceneObjects.end(); ++objsIter) {
			RENDERMANAGER->addToQueue ((*objsIter), m_MaterialMap);
		}
	}
	RENDERMANAGER->processQueue();	

}


void
Pass::restore(void) {

	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->unbind();
	}

	restoreCamera();
	RENDERER->removeLights();
}



// --------------------------------------------------
//		VIEWPORT
// --------------------------------------------------


void
Pass::setViewport(nau::render::Viewport *aViewport)
{
	m_Viewport = aViewport;
}


nau::render::Viewport *
Pass::getViewport()
{
	return (m_Viewport);
}



// --------------------------------------------------
//		LIGHTS
// --------------------------------------------------

void
Pass::addLight (const std::string &name) {
	assert(!hasLight(name));

	m_Lights.push_back (name);
}


void
Pass::removeLight(const std::string &name)
{
	std::vector <std::string>::iterator iter;

	iter = m_Lights.begin();
	while (iter != m_Lights.end() && (*iter) != name) {
		++iter;
	}
	if (iter != m_Lights.end())
		m_Lights.erase(iter);
}


bool
Pass::hasLight(const std::string &name) 
{
	std::vector <std::string>::iterator iter;
	iter = m_Lights.begin();

	while (iter != m_Lights.end() && (*iter) != name) {
		++iter;
	}
	if (iter != m_Lights.end())
		return true;
	else
		return false;
}


void
Pass::setupLights(void) {

	std::vector<std::string>::iterator lightsIter;
	lightsIter = m_Lights.begin();

	for (; lightsIter != m_Lights.end(); ++lightsIter) {
		Light *l = RENDERMANAGER->getLight(*lightsIter);
		RENDERER->addLight(*l);
	}
}


// --------------------------------------------------
//		MATERIAL MAPS
// --------------------------------------------------


const std::map<std::string, nau::material::MaterialID> &
Pass::getMaterialMap() {

	return m_MaterialMap;
}


void 
Pass::remapMaterial (const std::string &originMaterialName, const std::string &materialLib, const std::string &destinyMaterialName) {

	m_MaterialMap[originMaterialName].setMaterialID (materialLib, destinyMaterialName);
}


void 
Pass::remapAll (const std::string &materialLib, const std::string &destinyMaterialName) {

	m_RemapMode = REMAP_TO_ONE;
	m_MaterialMap["*"].setMaterialID (materialLib, destinyMaterialName);

	std::map<std::string, MaterialID>::iterator matMapIter;
	
	matMapIter = m_MaterialMap.begin();

	for ( ; matMapIter != m_MaterialMap.end(); ++matMapIter) {
		(*matMapIter).second.setMaterialID (materialLib, destinyMaterialName);
	}
}


void
Pass::remapAll (const std::string &targetLibrary) {

	m_RemapMode = REMAP_TO_LIBRARY;
	m_MaterialMap["*"].setMaterialID (targetLibrary, "*");

	std::map<std::string, MaterialID>::iterator matMapIter;
  
	matMapIter = m_MaterialMap.begin();
  
	for (; matMapIter != m_MaterialMap.end(); ++matMapIter) {
		if (MATERIALLIBMANAGER->hasMaterial(targetLibrary, (*matMapIter).first))
			(*matMapIter).second.setMaterialID (targetLibrary, (*matMapIter).first);
	}
}


void 
Pass::materialNamesFromLoadedScenes (std::vector<std::string> &materials) {

	std::vector<std::string>::iterator matIter;

	matIter = materials.begin();
	
	for ( ; matIter != materials.end(); ++matIter) {

		if (0 == m_MaterialMap.count (*matIter)) {

			switch (m_RemapMode) {
		  
				case REMAP_TO_ONE:
					m_MaterialMap[*matIter].setMaterialID (m_MaterialMap["*"].getLibName(), m_MaterialMap["*"].getMaterialName());
					break;

				case REMAP_TO_LIBRARY:
					m_MaterialMap[*matIter].setMaterialID (m_MaterialMap["*"].getLibName(), *matIter);
					break;

				default:
					m_MaterialMap[*matIter].setMaterialID (DEFAULTMATERIALLIBNAME, *matIter);
			}
		}
	}
}


void
Pass::updateMaterialMaps(const std::string &sceneName)
{
	if (this->hasScene(sceneName)) {
		std::set<std::string> *materialNames = new std::set<std::string>;
		RENDERMANAGER->getScene(sceneName)->getMaterialNames(materialNames);

		std::set<std::string>::iterator iter;
		iter = materialNames->begin();
		for (; iter != materialNames->end(); ++iter) {

			if (m_MaterialMap.count((*iter)) == 0)
				m_MaterialMap[(*iter)] = MaterialID(DEFAULTMATERIALLIBNAME, (*iter));
		}
		delete materialNames;
	}
}



// --------------------------------------------------
//		RENDER TARGETS
// --------------------------------------------------


bool
Pass::hasRenderTarget() {

	return (m_RenderTarget != 0);
}

 
nau::render::RenderTarget* 
Pass::getRenderTarget (void) {

	return m_RenderTarget;
}


bool 
Pass::isRenderTargetEnabled() {

	return m_UseRT;
}


void 
Pass::enableRenderTarget(bool b) {

	m_UseRT = b;
}


void 
Pass::setRenderTarget (nau::render::RenderTarget* rt) {
	
	if (rt == NULL) {
		if (m_RenderTarget != NULL) 
			delete m_Viewport;
		m_UseRT = true;
	}
	else {
		if (m_RenderTarget == NULL){
			m_Viewport = new Viewport();
			m_UseRT = true;
		}
		setRTSize(rt->getWidth(), rt->getHeight());
		m_Viewport->setPropf4(Viewport::CLEAR_COLOR, rt->getClearValues());
	}
	m_RenderTarget = rt;
}


void
Pass::setRTSize(int width, int height) {

	assert(m_Viewport != NULL);

	m_RTSizeWidth = width;
	m_RTSizeHeight = height;
	m_Viewport->setPropf2(Viewport::SIZE, vec2(width, height));
	m_Viewport->setPropf2(Viewport::ORIGIN, vec2(0.0f, 0.0f));
	m_Viewport->setPropb(Viewport::FULL, false);
}



// --------------------------------------------------
//		CAMERAS
// --------------------------------------------------


void
Pass::setupCamera (void) {

	Camera *aCam = 0;

	aCam = RENDERMANAGER->getCamera (m_CameraName);
	
	if (0 == aCam) {
		return; 
	}

	Viewport *v = aCam->getViewport();
	// if pass has a viewport 
	if (0 != m_Viewport ) {
		m_RestoreViewport = v;
		aCam->setViewport (m_Viewport);
	}
	
	RENDERER->setCamera (aCam);
}


void
Pass::restoreCamera (void) {

	Camera *aCam = 0;
	aCam = RENDERMANAGER->getCamera (m_CameraName);

	if (0 == aCam) {
		return; 
	}
	
	if (0 != m_Viewport ) {
		aCam->setViewport (m_RestoreViewport);
	}
}


const std::string& 
Pass::getCameraName (void) {

	return m_CameraName;
}


void 
Pass::setCamera (const std::string &cameraName) {

	m_CameraName = cameraName;
}



// --------------------------------------------------
//		SET PROPS
// --------------------------------------------------


void
Pass::setPropb(BoolProperty prop, bool value) {

	m_BoolProps[prop] = value;
}


void
Pass::setPropui(UIntProperty prop, unsigned int value) {

	m_UIntProps[prop] = value;
}


// --------------------------------------------------
//		STENCIL
// --------------------------------------------------


void 
Pass::setStencilFunc(StencilFunc f, int ref, unsigned int mask) {

	m_EnumProps[STENCIL_FUNC] = f;
	m_IntProps[STENCIL_OP_REF] = ref;
	m_UIntProps[STENCIL_OP_MASK] = mask;
}


void 
Pass::setStencilOp(	StencilOp sfail, StencilOp dfail, StencilOp dpass) {

	m_EnumProps[STENCIL_FAIL] = sfail;
	m_EnumProps[STENCIL_DEPTH_FAIL] = dfail;
	m_EnumProps[STENCIL_DEPTH_PASS] = dpass;
}


void 
Pass::setStencilClearValue(float v) {

	m_FloatProps[STENCIL_CLEAR_VALUE] = v;
}

// --------------------------------------------------
//		DEPTH
// --------------------------------------------------


void 
Pass::setDepthClearValue(float v) {

	m_FloatProps[DEPTH_CLEAR_VALUE] = v;
}


void 
Pass::setDepthFunc(int f) {

	m_EnumProps[DEPTH_FUNC] = f;
}



// --------------------------------------------------
//		SCENES
// --------------------------------------------------



void 
Pass::addScene (const std::string &sceneName) {

	if (m_SceneVector.end() == std::find (m_SceneVector.begin(), m_SceneVector.end(), sceneName)) {
	
		m_SceneVector.push_back (sceneName);
	
		std::set<std::string> *materialNames = new std::set<std::string>;
		RENDERMANAGER->getScene(sceneName)->getMaterialNames(materialNames);
		
		std::set<std::string>::iterator iter;
		iter = materialNames->begin();
		for ( ; iter != materialNames->end(); ++iter) {
			
			if (m_MaterialMap.count((*iter)) == 0)
				m_MaterialMap[(*iter)] = MaterialID(DEFAULTMATERIALLIBNAME, (*iter));
		}
		delete materialNames;
	}
//	RENDERMANAGER->getScene(sceneName)->compile();
}


void
Pass::removeScene(const std::string &name) {

	std::vector <std::string>::iterator iter;

	iter = m_SceneVector.begin();
	while (iter != m_SceneVector.end() && (*iter) != name) {
		++iter;
	}
	if (iter != m_SceneVector.end())
		m_SceneVector.erase(iter);
}


bool
Pass::hasScene(const std::string &name) {
	
	std::vector <std::string>::iterator iter;
	iter = m_SceneVector.begin();

	while (iter != m_SceneVector.end() && (*iter) != name) {
		++iter;
	}
	if (iter != m_SceneVector.end())
		return true;
	else
		return false;
}


const std::vector<std::string>& 
Pass::getScenesNames (void) {
	return m_SceneVector;
}


