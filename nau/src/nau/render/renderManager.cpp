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
	//m_pRenderer (0), 
	m_Pipelines(),
	m_Cameras(),
	m_Lights(),
//	m_ActivePipeline (0),
	m_RunMode(RUN_DEFAULT),
	m_DefaultCamera("__nauDefault") {

	m_pRenderer = std::unique_ptr<IRenderer>(RenderFactory::create());
	m_pRenderQueue = std::unique_ptr<IRenderQueue>(RenderQueueFactory::create ("MaterialSort"));
}


RenderManager::~RenderManager(void) {

}


void 
RenderManager::clear() {

	m_Cameras.clear();
	m_Lights.clear();
	m_Scenes.clear();
	m_Pipelines.clear();
	m_Viewports.clear();

	m_ActivePipelineIndex = 0;

	m_pRenderQueue->clearQueue();
}


bool 
RenderManager::init() {

	return(m_pRenderer->init());
}


// =========  VIEWPORTS  =========================


std::shared_ptr<Viewport> &
RenderManager::createViewport(const std::string &name, nau::math::vec4 &bgColor) {

	if (m_Viewports.count(name) == 0) {

		std::shared_ptr<Viewport> v = std::shared_ptr<Viewport>(new Viewport());

		v->setName(name);
		vec2 v2a(0.0f, 0.0f);
		v->setPropf2(Viewport::ORIGIN, v2a);
		vec2 v2b((float)NAU->getWindowWidth(), (float)NAU->getWindowHeight());
		v->setPropf2(Viewport::SIZE, v2b);

		v->setPropf4(Viewport::CLEAR_COLOR, bgColor);
		v->setPropb(Viewport::FULL, true);

		m_Viewports[name] = v;
	}	
	return m_Viewports[name];
}


bool
RenderManager::hasViewport(const std::string &name) {

	return (m_Viewports.count(name) != 0);
}


std::shared_ptr<Viewport> &
RenderManager::createViewport(const std::string &name) {

	if (m_Viewports.count(name) == 0) {
		std::shared_ptr<Viewport> v = std::shared_ptr<Viewport>(new Viewport());

		v->setName(name);
		//v->setPropf2(Viewport::ORIGIN, vec2(0.0f, 0.0f));
		//v->setPropf2(Viewport::SIZE, vec2((float)NAU->getWindowWidth(), (float)NAU->getWindowHeight()));
		v->setPropb(Viewport::FULL, true);

		m_Viewports[name] = v;
	}
	return m_Viewports[name];
}


std::shared_ptr<Viewport> 
RenderManager::getViewport(const std::string &name) {

	if (m_Viewports.count(name))
		return m_Viewports[name];
	else
		return nullptr;
}


void
RenderManager::getViewportNames(std::vector<std::string> *names) {

	for (auto &vp:m_Viewports) {
		names->push_back(vp.first);
	}
}


// =========  PIPELINES  =========================

std::shared_ptr<Pipeline> &
RenderManager::createPipeline(const std::string &pipelineName) {

	Pipeline *pip = new Pipeline(pipelineName);
	m_Pipelines.push_back(std::shared_ptr<Pipeline>(pip));

	return m_Pipelines[m_Pipelines.size()-1];
}


bool 
RenderManager::hasPipeline (const std::string &pipelineName) {

	for (auto p : m_Pipelines) {
		if (p->getName() == pipelineName)
			return true;
	}
	return false;
}


std::shared_ptr<Pipeline> &
RenderManager::getPipeline(const std::string &pipelineName) {

	for (auto &p : m_Pipelines) {
		if (p->getName() == pipelineName) 
			return p;
	}

	return m_Pipelines[0];
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


std::shared_ptr<Pipeline> &
RenderManager::getActivePipeline() {

	if (m_Pipelines.size() > m_ActivePipelineIndex)
		return m_Pipelines[m_ActivePipelineIndex];
	else
		return m_Pipelines[0];
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
RenderManager::setActivePipeline(int index) {

	m_ActivePipelineIndex = index;
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


void 
RenderManager::getPipelineNames(std::vector<std::string> *names) {

	for (auto p : m_Pipelines)
		names->push_back(p->getName());
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

	if (hasPipeline(pipeline)) {
		std::shared_ptr<Pipeline> &pip = getPipeline(pipeline);
		return pip->hasPass(pass);
	}
	else 
		return false;
}


Pass *RenderManager::getPass(const std::string &pipeline, const std::string &pass) {

	// Pipeline and pass must exist
	assert(hasPass(pipeline,pass));

	std::shared_ptr<Pipeline> &pip = getPipeline(pipeline);
	return pip->getPass(pass);
}


Pass *RenderManager::getPass(const std::string &pass) {

	assert(m_ActivePipelineIndex < m_Pipelines.size() && 
		m_Pipelines[m_ActivePipelineIndex]->hasPass(pass));

	// Pipeline and pass must exist
	std::shared_ptr<Pipeline> &active = m_Pipelines[m_ActivePipelineIndex];
	return active->getPass(pass);
}


Pass *
RenderManager::getCurrentPass() {

	assert(m_ActivePipelineIndex < m_Pipelines.size());
	return m_Pipelines[m_ActivePipelineIndex]->getCurrentPass();
}


void
RenderManager::prepareTriangleIDs(bool ids) {

	for (auto &s:m_Scenes) {

		std::vector <std::shared_ptr<SceneObject>> sceneObjs;
		s.second->getAllObjects(&sceneObjs);

		for (auto &so: sceneObjs ) {

			so->prepareTriangleIDs(ids);
		}

	}
}


void
RenderManager::renderActivePipelineNextPass() {

	if (m_ActivePipelineIndex < m_Pipelines.size())
		m_Pipelines[m_ActivePipelineIndex]->executeNextPass();
}


unsigned char
RenderManager::renderActivePipeline () 
{
	if (!(m_ActivePipelineIndex < m_Pipelines.size()))
		return 0;

	int n = RENDERER->getPropui(IRenderer::FRAME_COUNT);
	int k = m_Pipelines[m_ActivePipelineIndex]->getFrameCount();
	if (m_RunMode == RUN_ALL && k > 0 && k == n) {
		m_ActivePipelineIndex++;
		m_ActivePipelineIndex = m_ActivePipelineIndex % m_Pipelines.size();
		RENDERER->setPropui(IRenderer::FRAME_COUNT, 0);
		if (m_ActivePipelineIndex == 0)
			exit(0);
	}

	m_Pipelines[m_ActivePipelineIndex]->execute ();

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
RenderManager::pick (int x, int y, std::vector<std::shared_ptr<SceneObject>> &objects, nau::scene::Camera &aCamera) {

	return -1;
}


//void 
//RenderManager::setViewport(nau::render::Viewport* vp) {
//
//	m_pRenderer->setViewport(vp);
//}


void
RenderManager::setRenderMode (nau::render::IRenderer::TRenderMode mode) {

	m_pRenderer->setRenderMode (mode);
}


IRenderer*
RenderManager::getRenderer (void) {

	return m_pRenderer.get();
}


void
RenderManager::clearQueue (void) {

	m_pRenderQueue->clearQueue();
}


void 
RenderManager::addToQueue (std::shared_ptr<SceneObject> &aObject,
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


void
RenderManager::getCameraNames(std::vector<std::string> *names ) {

	for(auto &c:m_Cameras) {
      names->push_back(c.first); 
    }
}


const std::string&
RenderManager::getDefaultCameraName() {

	if (m_Pipelines.size() == 0)
		return m_DefaultCamera;

	// there must be an active pipeline
	//assert(m_ActivePipelineIndex < m_Pipelines.size());

	return(m_Pipelines[m_ActivePipelineIndex]->getDefaultCameraName());
}


std::shared_ptr<Camera> &
RenderManager::getCurrentCamera() {

	assert(m_ActivePipelineIndex < m_Pipelines.size());

	std::string cn = m_Pipelines[m_ActivePipelineIndex]->getCurrentCamera();
	return (m_Cameras[cn]);
}


std::shared_ptr<Camera> &
RenderManager::getCamera (const std::string &cameraName) {

	if (false == hasCamera (cameraName)) {
		m_Cameras[cameraName] = Camera::Create(cameraName);
	}
	return m_Cameras[cameraName];
}


std::shared_ptr<Camera> &
RenderManager::createCamera(const std::string &name) {

	if (m_Cameras.count(name) == 0) {
		std::shared_ptr<Camera> c = Camera::Create(name);

		c->setName(name);
		m_Cameras[name] = c;
	}
	return m_Cameras[name];
}

// -----------------------------------------------------------
//		LIGHTS
// -----------------------------------------------------------


unsigned int 
RenderManager::getNumLights() {

	return (unsigned int)m_Lights.size();
}


 void
RenderManager::getLightNames(std::vector<std::string> *names) {

	for( auto &light: m_Lights) {
      names->push_back(light.first); 
    }
}


bool 
RenderManager::hasLight (const std::string &lightName) {

	if (m_Lights.count (lightName) > 0) {
		return true;
	}
	return false;
}


std::shared_ptr<Light> &
RenderManager::getLight (const std::string &lightName) {

	if (false == hasLight (lightName)) {
		m_Lights[lightName] = std::shared_ptr<nau::scene::Light>(nau::scene::LightFactory::create(lightName, "Light"));
	}
	return m_Lights[lightName];
}


std::shared_ptr<Light> &
RenderManager::createLight (const std::string &lightName, const std::string &lightClass) {

	if (false == hasLight (lightName)) {
		m_Lights[lightName] = std::shared_ptr<nau::scene::Light>(nau::scene::LightFactory::create(lightName, lightClass));
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


void
RenderManager::getSceneNames(std::vector<std::string> *names) {

	std::map<std::string, std::shared_ptr<IScene>>::iterator iter = m_Scenes.begin();

	for( ; iter != m_Scenes.end(); ++iter ) {
		if ((*iter).second->getType() != "SceneAux")
      names->push_back((*iter).first); 
    }
}


void 
RenderManager::getAllSceneNames(std::vector<std::string> *names) {

	std::map<std::string, std::shared_ptr<IScene>>::iterator iter = m_Scenes.begin();

	for( ; iter != m_Scenes.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
}


void 
RenderManager::buildOctrees() {

	std::map<std::string, std::shared_ptr<IScene>>::iterator iter;
	for( iter = m_Scenes.begin(); iter != m_Scenes.end(); ++iter ) {
      ((*iter).second)->build(); 
    }
}


void 
RenderManager::compile() {

	std::map<std::string, std::shared_ptr<IScene>>::iterator iter;
	for( iter = m_Scenes.begin(); iter != m_Scenes.end(); ++iter ) {
      ((*iter).second)->compile(); 
    }
}


std::shared_ptr<IScene> &
RenderManager::createScene (const std::string &sceneName, const std::string &sceneType) {

	if (false == hasScene (sceneName)) {
		std::shared_ptr<IScene> s = SceneFactory::Create(sceneType);
		if (s) {
			m_Scenes[sceneName] = s;
			s->setName(sceneName);
		}
		return m_Scenes[sceneName];
	} 
	else
		return m_Scenes[sceneName]; //Or should it return NULL if it exists a scene with that name already
}


std::shared_ptr<IScene> &
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

