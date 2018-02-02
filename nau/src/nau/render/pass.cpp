#include "nau/render/pass.h"

#include "nau.h"
#include "nau/geometry/axis.h"
#include "nau/geometry/frustum.h"
#include "nau/render/passFactory.h"
#include "nau/render/opengl/glProfile.h"

#include <algorithm>
#include <sstream>

using namespace nau::material;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::geometry;


AttribSet Pass::Attribs;
bool Pass::Inited = Init();


bool
Pass::Init() {

	Attribs.add(Attribute(DIM_X, "DIM_X", Enums::DataType::UINT, false, new NauUInt(1)));
	Attribs.add(Attribute(DIM_Y, "DIM_Y", Enums::DataType::UINT, false, new NauUInt(1)));
	Attribs.add(Attribute(DIM_Z, "DIM_Z", Enums::DataType::UINT, false, new NauUInt(1)));

	// BOOL
	Attribs.add(Attribute(COLOR_CLEAR, "COLOR_CLEAR", Enums::DataType::BOOL, false, new NauInt(1)));
	Attribs.add(Attribute(COLOR_ENABLE, "COLOR_ENABLE", Enums::DataType::BOOL, false, new NauInt(1)));
	Attribs.add(Attribute(DEPTH_CLEAR, "DEPTH_CLEAR", Enums::DataType::BOOL, false, new NauInt(1)));
	Attribs.add(Attribute(DEPTH_ENABLE, "DEPTH_ENABLE", Enums::DataType::BOOL, false, new NauInt(1)));
	Attribs.add(Attribute(DEPTH_MASK, "DEPTH_MASK", Enums::DataType::BOOL, false, new NauInt(1)));
	Attribs.add(Attribute(DEPTH_CLAMPING, "DEPTH_CLAMPING", Enums::DataType::BOOL, false, new NauInt(0)));
	Attribs.add(Attribute(STENCIL_CLEAR, "STENCIL_CLEAR", Enums::DataType::BOOL, false, new NauInt(1)));
	Attribs.add(Attribute(STENCIL_ENABLE, "STENCIL_ENABLE", Enums::DataType::BOOL, false, new NauInt(0)));

	// ENUM
	Attribs.add(Attribute(TEST_MODE, "TEST_MODE", Enums::DataType::ENUM, true, new NauInt(RUN_IF)));
	Attribs.listAdd("TEST_MODE", "RUN_IF", RUN_IF);
	Attribs.listAdd("TEST_MODE", "RUN_WHILE", RUN_WHILE);

	Attribs.add(Attribute(RUN_MODE, "RUN_MODE", Enums::DataType::ENUM, true, new NauInt(RUN_ALWAYS)));
	Attribs.listAdd("RUN_MODE", "DONT_RUN", DONT_RUN);
	Attribs.listAdd("RUN_MODE", "RUN_ALWAYS", RUN_ALWAYS);
	Attribs.listAdd("RUN_MODE", "SKIP_FIRST_FRAME", SKIP_FIRST_FRAME);
	Attribs.listAdd("RUN_MODE", "RUN_ONCE", RUN_ONCE);
	Attribs.listAdd("RUN_MODE", "RUN_EVEN", RUN_EVEN);
	Attribs.listAdd("RUN_MODE", "RUN_ODD", RUN_ODD);

	Attribs.add(Attribute(STENCIL_FUNC, "STENCIL_FUNC", Enums::DataType::ENUM, false, new NauInt(ALWAYS)));
	Attribs.listAdd("STENCIL_FUNC", "LESS", LESS);
	Attribs.listAdd("STENCIL_FUNC", "NEVER", NEVER);
	Attribs.listAdd("STENCIL_FUNC", "ALWAYS", ALWAYS);
	Attribs.listAdd("STENCIL_FUNC", "LEQUAL", LEQUAL);
	Attribs.listAdd("STENCIL_FUNC", "EQUAL", EQUAL);
	Attribs.listAdd("STENCIL_FUNC", "GEQUAL", GEQUAL);
	Attribs.listAdd("STENCIL_FUNC", "GREATER", GREATER);
	Attribs.listAdd("STENCIL_FUNC", "NOT_EQUAL", NOT_EQUAL);

	Attribs.add(Attribute(STENCIL_FAIL, "STENCIL_FAIL", Enums::DataType::ENUM, false, new NauInt(KEEP)));
	Attribs.listAdd("STENCIL_FAIL", "KEEP", KEEP);
	Attribs.listAdd("STENCIL_FAIL", "ZERO", ZERO);
	Attribs.listAdd("STENCIL_FAIL", "REPLACE", REPLACE);
	Attribs.listAdd("STENCIL_FAIL", "INCR", INCR);
	Attribs.listAdd("STENCIL_FAIL", "INCR_WRAP", INCR_WRAP);
	Attribs.listAdd("STENCIL_FAIL", "DECR", DECR);
	Attribs.listAdd("STENCIL_FAIL", "DECR_WRAP", DECR_WRAP);
	Attribs.listAdd("STENCIL_FAIL", "INVERT", INVERT);

	Attribs.add(Attribute(STENCIL_DEPTH_FAIL, "STENCIL_DEPTH_FAIL", Enums::DataType::ENUM, false, new NauInt(KEEP)));
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "KEEP", KEEP);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "ZERO", ZERO);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "REPLACE", REPLACE);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "INCR", INCR);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "INCR_WRAP", INCR_WRAP);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "DECR", DECR);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "DECR_WRAP", DECR_WRAP);
	Attribs.listAdd("STENCIL_DEPTH_FAIL", "INVERT", INVERT);

	Attribs.add(Attribute(STENCIL_DEPTH_PASS, "STENCIL_DEPTH_PASS", Enums::DataType::ENUM, false, new NauInt(KEEP)));
	Attribs.listAdd("STENCIL_DEPTH_PASS", "KEEP", KEEP);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "ZERO", ZERO);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "REPLACE", REPLACE);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "INCR", INCR);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "INCR_WRAP", INCR_WRAP);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "DECR", DECR);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "DECR_WRAP", DECR_WRAP);
	Attribs.listAdd("STENCIL_DEPTH_PASS", "INVERT", INVERT);

	Attribs.add(Attribute(DEPTH_FUNC, "DEPTH_FUNC", Enums::DataType::ENUM, false, new NauInt(LESS)));
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
	Attribs.add(Attribute(DEPTH_CLEAR_VALUE, "DEPTH_CLEAR_VALUE", Enums::DataType::FLOAT, false, new NauFloat(1.0f)));

	//INT
	Attribs.add(Attribute(STENCIL_OP_REF, "STENCIL_OP_REF", Enums::DataType::INT, false, new NauInt(0)));
	Attribs.add(Attribute(STENCIL_CLEAR_VALUE, "STENCIL_CLEAR_VALUE", Enums::DataType::INT, false, new NauInt(0)));

	//UINT
	Attribs.add(Attribute(STENCIL_OP_MASK, "STENCIL_OP_MASK", Enums::DataType::UINT, false, new NauUInt(255)));
	Attribs.add(Attribute(INSTANCE_COUNT, "INSTANCE_COUNT", Enums::DataType::UINT, false, new NauUInt(0)));
	Attribs.add(Attribute(BUFFER_DRAW_INDIRECT, "BUFFER_DRAW_INDIRECT", Enums::DataType::UINT, true, new NauUInt(0)));

	//
	Attribs.add(Attribute(CAMERA, "camera", "CAMERA"));


	//#ifndef _WINDLL
	NAU->registerAttributes("PASS", &Attribs);
	//#endif

	PASSFACTORY->registerClass("default", Create);

	return true;
}


Pass::Pass (const std::string &passName) :
	m_ClassName("default"),
	m_Name (passName),
	//m_CameraName ("__nauDefault"),
	m_SceneVector(),
	m_MaterialMap(),
	m_Viewport (0),
	m_RestoreViewport (0),
	m_RemapMode (REMAP_DISABLED),
	m_TestScriptFile(""),
	m_BufferDrawIndirect(NULL),
	m_ExplicitViewport(false) {

	registerAndInitArrays(Attribs);

	initVars();
	//#ifndef _WINDLL
	EVENTMANAGER->addListener("SCENE_CHANGED",this);
	//#endif
}


AttribSet &
Pass::GetAttribs() {
	return Attribs;
}


Pass::~Pass() {

	// must delete pre and post process lists
}


std::shared_ptr<Pass>
Pass::Create(const std::string &passName) {

	return dynamic_pointer_cast<Pass>(std::shared_ptr<Pass>(new Pass(passName)));
}


void
Pass::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<IEventData> &evt)  {

	if (eventType == "SCENE_CHANGED") 
		updateMaterialMaps(sender);
}


const std::string &
Pass::getClassName() {

	return m_ClassName;
}


std::string &
Pass::getName (void) {

	return m_Name;
}



void 
Pass::initVars() {

	m_RenderTarget = NULL;

	m_RTSizeWidth = 512; // size of render targets
	m_RTSizeHeight = 512;

	m_UseRT = false; // enable, disable

	m_StringProps[CAMERA] = NAU->getActiveCamera()->getName();
}


// --------------------------------------------------
//		PROCESS ITEMS
// --------------------------------------------------

void 
Pass::addPreProcessItem(PassProcessItem *pp) {

	m_PreProcessList.push_back(pp);
}


void 
Pass::addPostProcessItem(PassProcessItem *pp) {

	m_PostProcessList.push_back(pp);
}


PassProcessItem *
Pass::getPreProcessItem(unsigned int i) {

	if (i < m_PreProcessList.size())
		return m_PreProcessList[i];
	else
		return NULL;
}


PassProcessItem *
Pass::getPostProcessItem(unsigned int i) {

	if (i < m_PostProcessList.size())
		return m_PostProcessList[i];
	else
		return NULL;
}


void
Pass::executePreProcessList() {

	PROFILE_GL("Pre Process List");
	for (auto pp : m_PreProcessList)
		pp->process();
}


void 
Pass::executePostProcessList() {

	PROFILE_GL("Post Process List");
	for (auto pp : m_PostProcessList)
		pp->process();
}

// --------------------------------------------------
//		BUFFER DRAW INDIRECT
// --------------------------------------------------

void 
Pass::setBufferDrawIndirect(std::string s) {

	m_BufferDrawIndirect = RESOURCEMANAGER->getBuffer(s);
	m_UIntProps[BUFFER_DRAW_INDIRECT] = m_BufferDrawIndirect->getPropi(IBuffer::ID);
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

#if NAU_LUA == 1
	bool test;
	if (m_TestScriptName != "") {
		test = NAU->callLuaTestScript(m_TestScriptName);
		if (!test)
			return false;
	}
#endif

	return true;
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

		if (m_ExplicitViewport) {
			vec2 f2 = m_Viewport->getPropf2(Viewport::ABSOLUTE_SIZE);
			m_RTSizeWidth = (int)f2.x;
			m_RTSizeHeight = (int)f2.y;
			uivec2 uiv2((unsigned int)m_RTSizeWidth, (unsigned int)m_RTSizeHeight);
			m_RenderTarget->setPropui2(IRenderTarget::SIZE, uiv2);
		}
		m_RenderTarget->bind();
	}

	setupCamera();
	setupLights();

	RENDERER->setPropui(IRenderer::INSTANCE_COUNT, m_UIntProps[INSTANCE_COUNT]);
	RENDERER->setPropui(IRenderer::BUFFER_DRAW_INDIRECT, m_UIntProps[BUFFER_DRAW_INDIRECT]);
}




void
Pass::doPass (void) {

	Frustum camFrustum;
	std::vector<SceneObject*>::iterator objsIter;
	std::vector<std::string>::iterator scenesIter;
	std::vector<std::shared_ptr<SceneObject>> sceneObjects;

	prepareBuffers();

	std::shared_ptr<Camera> &aCam = RENDERMANAGER->getCamera (m_StringProps[CAMERA]);
	const float *a = (float *)((mat4 *)RENDERER->getProp(IRenderer::PROJECTION_VIEW_MODEL, Enums::MAT4))->getMatrix();
	camFrustum.setFromMatrix (a);
	RENDERMANAGER->clearQueue();

	scenesIter = m_SceneVector.begin();

	for ( ; scenesIter != m_SceneVector.end(); ++scenesIter) {

		std::shared_ptr<IScene> &aScene = RENDERMANAGER->getScene (*scenesIter);
		{
			PROFILE("View Frustum Culling");
		//	aScene->findVisibleSceneObjects(&sceneObjects, camFrustum, *aCam);
			aScene->getAllObjects(&sceneObjects);
		}
		
	}
	for (auto &so: sceneObjects) {
		RENDERMANAGER->addToQueue (so, m_MaterialMap);
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
//		TEST SCRIPTS
// --------------------------------------------------


void 
Pass::setTestScript(std::string file, std::string name) {
#if NAU_LUA == 1
	m_TestScriptFile = file;
	m_TestScriptName = name;
	if (file != "" && name != "")
		NAU->initLuaScript(file, name);
#endif
}

// -----------------------------------------------------------------
//		PRE POST SCRIPTS
// -----------------------------------------------------------------

void 
Pass::setPreScript(std::string file, std::string name) {
#if NAU_LUA == 1
	m_PreScriptFile = file;
	m_PreScriptName = name;
	if (file != "" && name != "")
		NAU->initLuaScript(file, name);
#endif
}


void
Pass::setPostScript(std::string file, std::string name) {
#if NAU_LUA == 1
	m_PostScriptFile = file;
	m_PostScriptName = name;
	if (file != "" && name != "")
		NAU->initLuaScript(file, name);
#endif
}


void 
Pass::callScript(std::string &name) {

#if NAU_LUA == 1
	if (name != "") {
		NAU->callLuaScript(name);

	}
#endif
}


void 
Pass::callPreScript() {

	callScript(m_PreScriptName);
}


void 
Pass::callPostScript() {

	callScript(m_PostScriptName);
}



// --------------------------------------------------
//		VIEWPORT
// --------------------------------------------------


void
Pass::setViewport(std::shared_ptr<Viewport> aViewport) {

	m_Viewport = aViewport;
	m_ExplicitViewport = true;
}


std::shared_ptr<Viewport>
Pass::getViewport() {

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
Pass::removeLight(const std::string &name) {

	std::vector <std::string>::iterator iter;

	iter = m_Lights.begin();
	while (iter != m_Lights.end() && (*iter) != name) {
		++iter;
	}
	if (iter != m_Lights.end())
		m_Lights.erase(iter);
}


bool
Pass::hasLight(const std::string &name) {

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
		std::shared_ptr<Light> &l = RENDERMANAGER->getLight(*lightsIter);
		RENDERER->addLight(l);
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
Pass::updateMaterialMaps(const std::string &sceneName) {

	if (this->hasScene(sceneName)) {
		const std::set<std::string> &materialNames = 
			RENDERMANAGER->getScene(sceneName)->getMaterialNames();

		for (auto iter : materialNames) {

			if (m_MaterialMap.count(iter) == 0)
				m_MaterialMap[iter] = MaterialID(DEFAULTMATERIALLIBNAME, iter);
		}
	}
}



// --------------------------------------------------
//		RENDER TARGETS
// --------------------------------------------------


bool
Pass::hasRenderTarget() {

	return (m_RenderTarget != 0);
}

 
nau::render::IRenderTarget* 
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
Pass::setRenderTarget (nau::render::IRenderTarget* rt) {
	
	if (rt == NULL) {
		if (m_RenderTarget != NULL && !m_ExplicitViewport) 
			m_Viewport.reset();
		m_UseRT = true;
	}
	else {
		if (m_RenderTarget == NULL){
			std::string s = "__" + m_Name;
			m_Viewport = RENDERMANAGER->createViewport(s);
			m_UseRT = true;
		}
		setRTSize(rt->getPropui2(IRenderTarget::SIZE));
		m_Viewport->setPropf4(Viewport::CLEAR_COLOR, rt->getPropf4(IRenderTarget::CLEAR_VALUES));
	}
	m_RenderTarget = rt;
}


void
Pass::setRTSize(uivec2 &v) {

	assert(m_Viewport != NULL);

	m_RTSizeWidth = v.x;
	m_RTSizeHeight = v.y;
	vec2 v2a((float)v.x, (float)v.y);
	m_Viewport->setPropf2(Viewport::SIZE, v2a);
	vec2 v2b(0.0f, 0.0f);
	m_Viewport->setPropf2(Viewport::ORIGIN, v2b);
	m_Viewport->setPropb(Viewport::FULL, false);
}



// --------------------------------------------------
//		CAMERAS
// --------------------------------------------------


void
Pass::setupCamera (void) {

	std::shared_ptr<Camera> &aCam = RENDERMANAGER->getCamera (m_StringProps[CAMERA]);
	
	std::shared_ptr<Viewport> v = aCam->getViewport();
	// if pass has a viewport 
	if (m_Viewport) {
		m_RestoreViewport = v;
		aCam->setViewport (m_Viewport);
	}
	
	RENDERER->setCamera (aCam);
}


void
Pass::restoreCamera (void) {

	std::shared_ptr<Camera> &aCam = RENDERMANAGER->getCamera(m_StringProps[CAMERA]);

	if (m_ExplicitViewport) {
		aCam->setViewport (m_RestoreViewport);
	}
}


const std::string& 
Pass::getCameraName (void) {

	return m_StringProps[CAMERA];
}


void 
Pass::setCamera (const std::string &cameraName) {

	m_StringProps[CAMERA] = cameraName;
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
Pass::setStencilClearValue(unsigned int v) {

	m_IntProps[STENCIL_CLEAR_VALUE] = v;
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
	
		const std::set<std::string> &materialNames =
			RENDERMANAGER->getScene(sceneName)->getMaterialNames();
		
		for (auto name: materialNames) {
			if (m_MaterialMap.count("*") != 0) {
				m_MaterialMap[name] = MaterialID(m_MaterialMap["*"].getLibName(), m_MaterialMap["*"].getMaterialName());
			}
			else if (m_MaterialMap.count(name) == 0)
				m_MaterialMap[name] = MaterialID(DEFAULTMATERIALLIBNAME, name);
		}
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


