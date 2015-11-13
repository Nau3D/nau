#include "nau/render/renderManager.h"

#include "nau.h"
#include "nau/math/vec3.h"
#include "nau/render/iRenderer.h"
#include "nau/render/renderFactory.h"
#include "nau/render/renderQueueFactory.h"
#include "nau/scene/iScene.h"
#include "nau/scene/lightFactory.h"
#include "nau/scene/sceneFactory.h"

using namespace nau::render;
using namespace nau::scene;
using namespace nau::material;
using namespace nau::math;


RenderManager::RenderManager(void) : 
	m_pRenderer (0), 
	m_Pipelines(),
	m_Cameras(),
	m_Lights(),
//	m_ActivePipeline (0),
	m_RunMode(RUN_DEFAULT) {

	m_pRenderer = RenderFactory::create();
	m_pRenderQueue = RenderQueueFactory::create ("MaterialSort");
}


RenderManager::~RenderManager(void) {

	clear();
	delete m_pRenderer;
	delete m_pRenderQueue;
}


void 
RenderManager::clear() {

	while (!m_Cameras.empty()){
		delete ((*m_Cameras.begin()).second);
		m_Cameras.erase(m_Cameras.begin());
	}

	while (!m_Lights.empty()){
		delete ((*m_Lights.begin()).second);
		m_Lights.erase(m_Lights.begin());
	}

	while (!m_Scenes.empty()){
		delete ((*m_Scenes.begin()).second);
		m_Scenes.erase(m_Scenes.begin());
	}

	while (!m_Pipelines.empty()){
		delete ((*m_Pipelines.begin()));
		m_Pipelines.erase(m_Pipelines.begin());
	}
	while (!m_Viewports.empty()){
		delete ((*m_Viewports.begin()).second);
		m_Viewports.erase(m_Viewports.begin());
	}


	m_ActivePipelineIndex = 0;

	m_pRenderQueue->clearQueue();

	//while (!m_SceneObjects.empty()){
	//	delete (*m_SceneObjects.begin());
	//	m_SceneObjects.erase(m_SceneObjects.begin());
	//}
}


bool 
RenderManager::init() {

	return(m_pRenderer->init());
}


// =========  VIEWPORTS  =========================


Viewport*
RenderManager::createViewport(const std::string &name, nau::math::vec4 &bgColor) {

	Viewport* v = new Viewport;

	v->setName(name);
	v->setPropf2(Viewport::ORIGIN, vec2(0.0f, 0.0f));
	v->setPropf2(Viewport::SIZE, vec2((float)NAU->getWindowWidth(), (float)NAU->getWindowHeight()));

	v->setPropf4(Viewport::CLEAR_COLOR, bgColor);
	v->setPropb(Viewport::FULL, true);

	m_Viewports[name] = v;

	return v;
}


bool
RenderManager::hasViewport(const std::string &name) {

	return (m_Viewports.count(name) != NULL);
}


Viewport*
RenderManager::createViewport(const std::string &name) {

	Viewport* v = new Viewport;

	v->setName(name);
	v->setPropf2(Viewport::ORIGIN, vec2(0.0f, 0.0f));
	v->setPropf2(Viewport::SIZE, vec2((float)NAU->getWindowWidth(), (float)NAU->getWindowHeight()));
	v->setPropb(Viewport::FULL, true);

	m_Viewports[name] = v;

	return v;
}


Viewport*
RenderManager::getViewport(const std::string &name) {

	if (m_Viewports.count(name))
		return m_Viewports[name];
	else
		return NULL;
}


std::vector<std::string> *
RenderManager::getViewportNames() {

	std::vector<std::string> *names = new std::vector<std::string>;

	for (std::map<std::string, nau::render::Viewport*>::iterator iter = m_Viewports.begin(); iter != m_Viewports.end(); ++iter) {
		names->push_back(iter->first);
	}
	return names;
}


// =========  PIPELINES  =========================

bool 
RenderManager::hasPipeline (const std::string &pipelineName) {

	for (auto p : m_Pipelines) {
		if (p->getName() == pipelineName)
			return true;
	}
	return false;
}


Pipeline*
RenderManager::getPipeline(const std::string &pipelineName) {

	Pipeline *pip = NULL;
	bool found = false;

	for (auto p : m_Pipelines) {
		if (p->getName() == pipelineName) {
			pip = p;
			found = true;
			break;
		}
	}

	if (!found) {
		pip = new Pipeline(pipelineName);
		m_Pipelines.push_back(pip);
	}

	return pip;
}


unsigned int
RenderManager::getPipelineIndex(const std::string &pipelineName) {

	int index = 0;
	bool found = false;

	for (auto p : m_Pipelines) {
		if (p->getName() == pipelineName) {
			found = true;
			break;
		}
		index++;
	}

	if (!found) {
		index = 0;
	}

	return index;
}


Pipeline*
RenderManager::getActivePipeline() {

	if (m_Pipelines.size() > m_ActivePipelineIndex)
		return m_Pipelines[m_ActivePipelineIndex];
	else
		return NULL;
}
 

std::string
RenderManager::getActivePipelineName() {

	if (m_Pipelines.size() > m_ActivePipelineIndex) {
		return m_Pipelines[m_ActivePipelineIndex]->getName();
	}
	else
		return "";
}


void
RenderManager::setActivePipeline (const std::string &pipelineName) {

	m_ActivePipelineIndex = getPipelineIndex(pipelineName);
}


void
RenderManager::setActivePipeline (unsigned int index) {

	if (index < m_Pipelines.size()) {
		m_ActivePipelineIndex = index;
	}
}


unsigned int 
RenderManager::getNumPipelines() {

	return (unsigned int)m_Pipelines.size();
}


std::vector<std::string> * 
RenderManager::getPipelineNames() {

	std::vector<std::string> *names = new std::vector<std::string>; 

	for (auto p : m_Pipelines)
		names->push_back(p->getName());

	return names;
}


bool
RenderManager::setRunMode(std::string mode) {

	if (mode == "RUN_ALL")
		m_RunMode = RUN_ALL;
	else if (mode == "RUN_DEFAULT")
		m_RunMode = RUN_DEFAULT;
	else
		return false;

	return true;
}


// =========  PASS  =========================

bool
RenderManager::hasPass(const std::string &pipeline, const std::string &pass) {

	Pipeline *pip;

	if (hasPipeline(pipeline)) {
		pip = getPipeline(pipeline);
		return pip->hasPass(pass);
	}
	else 
		return false;
}


Pass *RenderManager::getPass(const std::string &pipeline, const std::string &pass) {

	// Pipeline and pass must exist
	assert(hasPass(pipeline,pass));

	Pipeline *pip = getPipeline(pipeline);
	return pip->getPass(pass);
}


Pass *RenderManager::getPass(const std::string &pass) {

	assert(m_ActivePipelineIndex < m_Pipelines.size() && 
		m_Pipelines[m_ActivePipelineIndex]->hasPass(pass));

	// Pipeline and pass must exist
	Pipeline *active = m_Pipelines[m_ActivePipelineIndex];
	return active->getPass(pass);
}


Pass *
RenderManager::getCurrentPass() {

	assert(m_ActivePipelineIndex < m_Pipelines.size());
	return m_Pipelines[m_ActivePipelineIndex]->getCurrentPass();
}


Camera*
RenderManager::getCurrentCamera() {

	assert(m_ActivePipelineIndex < m_Pipelines.size());

	std::string cn = m_Pipelines[m_ActivePipelineIndex]->getCurrentCamera();
	return (m_Cameras[cn]);
}


void
RenderManager::prepareTriangleIDs(bool ids) {

	std::map<std::string, nau::scene::IScene*>::iterator sceneIter;

	sceneIter = m_Scenes.begin();
	for ( ; sceneIter != m_Scenes.end(); ++sceneIter) {

		std::vector <SceneObject*> sceneObjs = (*sceneIter).second->getAllObjects();

		std::vector <SceneObject*>::iterator sceneObjIter;

		sceneObjIter = sceneObjs.begin();
		for ( ; sceneObjIter != sceneObjs.end(); ++sceneObjIter ) {

			(*sceneObjIter)->prepareTriangleIDs(ids);
		}

	}

	//if (ids) {
	//	int total = SceneObject::Counter;
	//	m_SceneObjects.resize(total);
	//	sceneIter = m_Scenes.begin();
	//	int count = 0;
	//	for ( ; sceneIter != m_Scenes.end(); sceneIter++) {

	//		std::vector <SceneObject*> sceneObjs = (*sceneIter).second->getAllObjects();

	//		std::vector <SceneObject*>::iterator sceneObjIter;

	//		sceneObjIter = sceneObjs.begin();
	//		for ( ; sceneObjIter != sceneObjs.end(); sceneObjIter ++) {
	//			if ((*sceneObjIter)->getId() != 0) {
	//				count++;
	//				m_SceneObjects[(*sceneObjIter)->getId()-1] = (*sceneObjIter);
	//			}
	//		}
	//	}
	//	m_SceneObjects.resize(count);
	//}

}

SceneObject *
RenderManager::getSceneObject(int id) {

	std::vector<nau::scene::SceneObject*>::iterator iter;
	for (iter = m_SceneObjects.begin(); iter != m_SceneObjects.end() && (*iter)->getId() != id; ++iter);

	if (iter != m_SceneObjects.end())
		return *iter;
	else
		return NULL;
}


void 
RenderManager::addSceneObject(SceneObject *s) {

	m_SceneObjects.push_back(s);
}


void 
RenderManager::deleteSceneObject(int id) {

	std::vector<nau::scene::SceneObject*>::iterator iter;

	for (iter = m_SceneObjects.begin(); iter != m_SceneObjects.end() && (*iter)->getId() != id; ++iter);

	if (iter != m_SceneObjects.end())
		m_SceneObjects.erase(iter);
}


void
RenderManager::getVertexData(unsigned int sceneObjID, unsigned int triID) {


}


void
RenderManager::renderActivePipelineNextPass() {

	if (m_ActivePipelineIndex < m_Pipelines.size())
		m_Pipelines[m_ActivePipelineIndex]->executeNextPass();
}


unsigned char
RenderManager::renderActivePipeline () 
{
	Pipeline *pip;

	if (!(m_ActivePipelineIndex < m_Pipelines.size()))
		return 0;

	pip = m_Pipelines[m_ActivePipelineIndex];

	int n = RENDERER->getPropui(IRenderer::FRAME_COUNT);
	int k = pip->getFrameCount();
	if (m_RunMode == RUN_ALL && k > 0 && k == n) {
		m_ActivePipelineIndex++;
		m_ActivePipelineIndex = m_ActivePipelineIndex % m_Pipelines.size();
		NAU->resetFrameCount();
		if (m_ActivePipelineIndex == 0)
			exit(0);
	}

	pip = m_Pipelines[m_ActivePipelineIndex];
	pip->execute ();

	return 0;
}


void *
RenderManager::getCurrentPassAttribute(std::string name, Enums::DataType dt) {

	assert(m_ActivePipelineIndex < m_Pipelines.size());

	int id = Pass::Attribs.getID(name);
	return m_Pipelines[m_ActivePipelineIndex]->getCurrentPass()->getProp(id, dt);
}


Enums::DataType
RenderManager::getPassAttributeType(std::string name) {

	Enums::DataType dt = Pass::Attribs.get(name)->getType();
	return dt;
}


//float *
//RenderManager::getPassParamf(std::string passName, std::string paramName) {
//
//	return(m_ActivePipeline->getPass(passName)->getParamf(paramName));
//}
void *
RenderManager::getPassAttribute(std::string passName, std::string name, Enums::DataType dt) {

	int id = Pass::Attribs.getID(name);
	return m_Pipelines[m_ActivePipelineIndex]->getPass(passName)->getProp(id, dt);
}




int
RenderManager::pick (int x, int y, std::vector<nau::scene::SceneObject*> &objects, nau::scene::Camera &aCamera) {

	return -1;
}


void 
RenderManager::setViewport(nau::render::Viewport* vp) {

	m_pRenderer->setViewport(vp);
}


void
RenderManager::setRenderMode (nau::render::IRenderer::TRenderMode mode) {

	m_pRenderer->setRenderMode (mode);
}


IRenderer*
RenderManager::getRenderer (void) {

	return m_pRenderer;
}


void
RenderManager::clearQueue (void) {

	m_pRenderQueue->clearQueue();
}


void 
RenderManager::addToQueue (SceneObject *aObject, 
				std::map<std::string, MaterialID> &materialMap) {

	m_pRenderQueue->addToQueue (aObject, materialMap);
}


void 
RenderManager::processQueue (void) {
	m_pRenderQueue->processQueue();
}


// -----------------------------------------------------------
//		CAMERAS
// -----------------------------------------------------------

bool
RenderManager::hasCamera (const std::string &cameraName) {

	if (m_Cameras.count (cameraName) > 0) {
		return true;
	}
	return false;
}


unsigned int 
RenderManager::getNumCameras() {

	return (unsigned int)m_Cameras.size();
}


std::vector<std::string> * 
RenderManager::getCameraNames() {

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::map<std::string, nau::scene::Camera*>::iterator iter = m_Cameras.begin(); iter != m_Cameras.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
	return names;
}


const std::string&
RenderManager::getDefaultCameraName() {

	// there must be an active pipeline
	assert(m_ActivePipelineIndex < m_Pipelines.size());

	return(m_Pipelines[m_ActivePipelineIndex]->getDefaultCameraName());
}


Camera* 
RenderManager::getCamera (const std::string &cameraName) {
	if (false == hasCamera (cameraName)) {
		m_Cameras[cameraName] = new Camera (cameraName);
	}
	return m_Cameras[cameraName];
}

// -----------------------------------------------------------
//		LIGHTS
// -----------------------------------------------------------


unsigned int 
RenderManager::getNumLights() {

	return (unsigned int)m_Lights.size();
}


std::vector<std::string> * 
RenderManager::getLightNames() {

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::map<std::string, nau::scene::Light*>::iterator iter = m_Lights.begin(); iter != m_Lights.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
	return names;
}


bool 
RenderManager::hasLight (const std::string &lightName) {

	if (m_Lights.count (lightName) > 0) {
		return true;
	}
	return false;
}


Light* 
RenderManager::getLight (const std::string &lightName) {

	if (false == hasLight (lightName)) {
		m_Lights[lightName] = nau::scene::LightFactory::create(lightName,"Light");
	}
	return m_Lights[lightName];
}


Light* 
RenderManager::getLight (const std::string &lightName, const std::string &lightClass) {

	if (false == hasLight (lightName)) {
		m_Lights[lightName] = nau::scene::LightFactory::create(lightName,lightClass);
	}
	return m_Lights[lightName];
}


// -----------------------------------------------------------
//		SCENES
// -----------------------------------------------------------


bool 
RenderManager::hasScene (const std::string &sceneName) {
	if (m_Scenes.count (sceneName) > 0) {
		return true;
	}
	return false;
}


std::vector<std::string> * 
RenderManager::getSceneNames() {

	std::vector<std::string> *names = new std::vector<std::string>; 

	std::map<std::string, nau::scene::IScene*>::iterator iter = m_Scenes.begin();

	for( ; iter != m_Scenes.end(); ++iter ) {
		if ((*iter).second->getType() != "SceneAux")
      names->push_back((*iter).first); 
    }
	return names;
}


std::vector<std::string> * 
RenderManager::getAllSceneNames() {

	std::vector<std::string> *names = new std::vector<std::string>; 

	std::map<std::string, nau::scene::IScene*>::iterator iter = m_Scenes.begin();

	for( ; iter != m_Scenes.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
	return names;
}


void 
RenderManager::buildOctrees() {

	std::map<std::string, nau::scene::IScene*>::iterator iter;
	for( iter = m_Scenes.begin(); iter != m_Scenes.end(); ++iter ) {
      ((*iter).second)->build(); 
    }
}


void 
RenderManager::compile() {

	std::map<std::string, nau::scene::IScene*>::iterator iter;
	for( iter = m_Scenes.begin(); iter != m_Scenes.end(); ++iter ) {
      ((*iter).second)->compile(); 
    }
}


nau::scene::IScene* 
RenderManager::createScene (const std::string &sceneName, const std::string &sceneType) {

	if (false == hasScene (sceneName)) {
		IScene *s = SceneFactory::create (sceneType);
		if (s) {
			m_Scenes[sceneName] = s;
			s->setName(sceneName);
		}
	} 
	return m_Scenes[sceneName]; //Or should it return NULL if it exists a scene with that name already
}


nau::scene::IScene* 
RenderManager::getScene (const std::string &sceneName) {

	if (false == hasScene (sceneName)) {
		createScene (sceneName);
	}
	return m_Scenes[sceneName];
}


void 
RenderManager::materialNamesFromLoadedScenes (std::vector<std::string> &materials) {

	for (auto p : m_Pipelines) {

		int passCount = p->getNumberOfPasses();
		for (int i = 0; i < passCount; i++) {
			p->getPass (i)->materialNamesFromLoadedScenes (materials);
		}

	}
}

