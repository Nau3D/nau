#include <nau/render/rendermanager.h>

#include <nau/render/irenderer.h>
#include <nau/render/renderfactory.h>
#include <nau/render/renderqueuefactory.h>
#include <nau/scene/iscene.h>
#include <nau/scene/scenefactory.h>
#include <nau/math/vec3.h>

#include <nau/scene/lightFactory.h>

using namespace nau::render;
using namespace nau::scene;
using namespace nau::material;
using namespace nau::math;


RenderManager::RenderManager(void) : 
	m_pRenderer (0), 
	m_Pipelines(),
	m_Cameras(),
	m_Lights(),
	m_ActivePipeline (0)
{
	m_pRenderer = RenderFactory::create();
	m_pRenderQueue = RenderQueueFactory::create ("MaterialSort");
}

RenderManager::~RenderManager(void)
{
	clear();
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
		delete ((*m_Pipelines.begin()).second);
		m_Pipelines.erase(m_Pipelines.begin());
	}

	m_ActivePipeline = 0;

	m_pRenderQueue->clearQueue();

//	m_SceneObjects.clear();
	while (!m_SceneObjects.empty()){
		delete ((*m_SceneObjects.begin()));
		m_SceneObjects.erase(m_SceneObjects.begin());
	}


}

bool RenderManager::init() {

	return(m_pRenderer->init());
}


// =========  PIPELINES  =========================

bool 
RenderManager::hasPipeline (const std::string &pipelineName)
{
	return (m_Pipelines.count (pipelineName) > 0);
}

Pipeline*
RenderManager::getPipeline (const std::string &pipelineName)
{
	if (m_Pipelines.find (pipelineName) == m_Pipelines.end()) {
		m_Pipelines[pipelineName] = new Pipeline (pipelineName);
		if (m_Pipelines.size() == 1)
			m_ActivePipeline = m_Pipelines[pipelineName];
	}
	return m_Pipelines[pipelineName];
}

void
RenderManager::setActivePipeline (const std::string &pipelineName)
{
	m_ActivePipeline = m_Pipelines[pipelineName];
}

unsigned int 
RenderManager::getNumPipelines() {

	return m_Pipelines.size();
}

std::vector<std::string> * 
RenderManager::getPipelineNames(){

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::map<std::string, nau::render::Pipeline*>::iterator iter = m_Pipelines.begin(); iter != m_Pipelines.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
	return names;
}



bool
RenderManager::hasPass(const std::string &pipeline, const std::string &pass) 
{
	if (m_Pipelines.count(pipeline)) {

		if (m_Pipelines[pipeline]->hasPass(pass))
			return true;
		else
			return false;
	}
	else
		return false;
}

Pass *RenderManager::getPass(const std::string &pipeline, const std::string &pass)
{
	// Pipeline and pass must exist
	assert(hasPass(pipeline,pass));

	return m_Pipelines[pipeline]->getPass(pass);
}

Camera*
RenderManager::getCurrentCamera() {

	std::string cn = m_ActivePipeline->getCurrentCamera();

	return (m_Cameras[cn]);
}

void
RenderManager::prepareTriangleIDs(bool ids) 
{
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
RenderManager::addSceneObject(SceneObject *s) 
{
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
RenderManager::renderActivePipeline () 
{
	// There must be an active pipeline
	// assert(m_ActivePipeline);

	if (m_ActivePipeline)
		m_ActivePipeline->execute ();
}


float *
RenderManager::getCurrentPassParamf(std::string name) {

	return(m_ActivePipeline->getCurrentPass()->getParamf(name));
}


int
RenderManager::getCurrentPassParamType(std::string name) {

	return(m_ActivePipeline->getCurrentPass()->getParamType(name));
}


float *
RenderManager::getPassParamf(std::string passName, std::string paramName) {

	return(m_ActivePipeline->getPass(passName)->getParamf(paramName));
}


int
RenderManager::getPassParamType(std::string passName, std::string paramName) {

	return(m_ActivePipeline->getPass(passName)->getParamType(paramName));
}


int
RenderManager::pick (int x, int y, std::vector<nau::scene::SceneObject*> &objects, nau::scene::Camera &aCamera)
{
	//m_pRenderer->setMatrix (IRenderer::PROJECTION);
	//m_pRenderer->push();
	//m_pRenderer->loadIdentity();

	//m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	//m_pRenderer->push();
	//m_pRenderer->loadIdentity();

	//m_pRenderer->setCamera (aCamera);
	//m_pRenderer->clear (IRenderer::COLORBUFFER | IRenderer::DEPTHBUFFER);

	//m_pRenderer->disableSurfaceShaders();
	//m_pRenderer->deactivateLighting();
	//m_pRenderer->disableTexturing();

	//std::vector<nau::scene::ISceneObject*>::iterator objIter;

	//objIter = objects.begin();

	//int i, j;

	//for (i = 10; objIter != objects.end(); objIter++, i+=10) {
	//	m_pRenderer->setColor (i / 255.0f, 0.0f, 0.0f, 1.0f);
	//	m_pRenderer->renderObject (*(*objIter), VertexData::DRAW_VERTICES);
	//}

	//m_pRenderer->setColor (1.0f, 1.0f, 1.0f, 1.0f);
	//
	//m_pRenderer->enableSurfaceShaders();

	//m_pRenderer->setMatrix (IRenderer::PROJECTION);
	//m_pRenderer->pop();

	//m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	//m_pRenderer->pop();

	//vec3 result = m_pRenderer->readpixel (x, aCamera.getViewport().getSize().y - y);

	//for (i = 10, j = 0; j < objects.size(); i+=10, j++){
	//	if (result.x == i) {
	//		return j;
	//	}
	//}

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


//void 
//RenderManager::enableStereo (void)
//{
//	m_pRenderer->enableStereo();
//}
//
//void 
//RenderManager::disableStereo (void)
//{
//	m_pRenderer->disableStereo();
//}

IRenderer*
RenderManager::getRenderer (void)
{
	return m_pRenderer;
}

void
RenderManager::clearQueue (void)
{
	m_pRenderQueue->clearQueue();
}

void 
RenderManager::addToQueue (SceneObject *aObject, 
				std::map<std::string, MaterialID> &materialMap)
{
	m_pRenderQueue->addToQueue (aObject, materialMap);
}

void 
RenderManager::processQueue (void)
{
	m_pRenderQueue->processQueue();
}

bool 
RenderManager::hasCamera (const std::string &cameraName)
{
	if (m_Cameras.count (cameraName) > 0) {
		return true;
	}
	return false;
}

unsigned int 
RenderManager::getNumCameras() {

	return m_Cameras.size();
}

std::vector<std::string> * 
RenderManager::getCameraNames(){

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::map<std::string, nau::scene::Camera*>::iterator iter = m_Cameras.begin(); iter != m_Cameras.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
	return names;
}




const std::string&
RenderManager::getDefaultCameraName() {

	// there must be an active pipeline
	assert(m_ActivePipeline != NULL);

	return(m_ActivePipeline->getDefaultCameraName());
}



Camera* 
RenderManager::getCamera (const std::string &cameraName)
{
	if (false == hasCamera (cameraName)) {
		m_Cameras[cameraName] = new Camera (cameraName);
	}
	return m_Cameras[cameraName];
}

// LIGHTS

unsigned int 
RenderManager::getNumLights() {

	return m_Lights.size();
}

std::vector<std::string> * 
RenderManager::getLightNames(){

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::map<std::string, nau::scene::Light*>::iterator iter = m_Lights.begin(); iter != m_Lights.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
	return names;
}
bool 
RenderManager::hasLight (const std::string &lightName)
{
	if (m_Lights.count (lightName) > 0) {
		return true;
	}
	return false;
}

Light* 
RenderManager::getLight (const std::string &lightName)
{
	if (false == hasLight (lightName)) {
		m_Lights[lightName] = nau::scene::LightFactory::create(lightName,"Light");
	}
	return m_Lights[lightName];
}

Light* 
RenderManager::getLight (const std::string &lightName, const std::string &lightClass)
{
	if (false == hasLight (lightName)) {
		m_Lights[lightName] = nau::scene::LightFactory::create(lightName,lightClass);
	}
	return m_Lights[lightName];
}


bool 
RenderManager::hasScene (const std::string &sceneName)
{
	if (m_Scenes.count (sceneName) > 0) {
		return true;
	}
	return false;
}


std::vector<std::string> * 
RenderManager::getSceneNames()
{
	std::vector<std::string> *names = new std::vector<std::string>; 

	std::map<std::string, nau::scene::IScene*>::iterator iter = m_Scenes.begin();

	for( ; iter != m_Scenes.end(); ++iter ) {
		if ((*iter).second->getType() != "SceneAux")
      names->push_back((*iter).first); 
    }
	return names;
}


std::vector<std::string> * 
RenderManager::getAllSceneNames()
{
	std::vector<std::string> *names = new std::vector<std::string>; 

	std::map<std::string, nau::scene::IScene*>::iterator iter = m_Scenes.begin();

	for( ; iter != m_Scenes.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
	return names;
}



void 
RenderManager::buildOctrees()
{
	std::map<std::string, nau::scene::IScene*>::iterator iter;
	for( iter = m_Scenes.begin(); iter != m_Scenes.end(); ++iter ) {
      ((*iter).second)->build(); 
    }
}


void 
RenderManager::compile()
{
	std::map<std::string, nau::scene::IScene*>::iterator iter;
	for( iter = m_Scenes.begin(); iter != m_Scenes.end(); ++iter ) {
      ((*iter).second)->compile(); 
    }
}


nau::scene::IScene* 
RenderManager::createScene (const std::string &sceneName, const std::string &sceneType)
{
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
RenderManager::getScene (const std::string &sceneName)
{
	if (false == hasScene (sceneName)) {
		createScene (sceneName);
	}
	return m_Scenes[sceneName];
}



void 
RenderManager::materialNamesFromLoadedScenes (std::vector<std::string> &materials)
{
	std::map<std::string, Pipeline*>::iterator pipIter;

	pipIter = m_Pipelines.begin();

	for ( ; pipIter != m_Pipelines.end(); pipIter++) {
		int passCount = (*pipIter).second->getNumberOfPasses();
		for (int i = 0; i < passCount; i++) {
			(*pipIter).second->getPass (i)->materialNamesFromLoadedScenes (materials);
		}
	}
}

/* This method returns the renderer instance.
For now the chosen implementation of the IRenderer interface is hard-coded.
*/

//RenderManager&
//RenderManager::getInstance ()
//{
//	if (0 == s_Instance){
//		s_Instance = new RenderManager;
//
//		s_Instance->m_pRenderer = RenderFactory::create();
//	}
//	return *s_Instance;	
//}

//void
//RenderManager::addAlgorithm (std::string algorithm) {
//	
//	IRenderAlgorithm *aAlgo = AlgorithmFactory::create (algorithm);
//
//	m_vRenderAlgorithms.push_back (aAlgo);
//	aAlgo->setRenderer (m_pRenderer);
//	aAlgo->init();
//}

//void
//RenderManager::reload (void) {
//	//std::vector<IRenderAlgorithm*>::iterator algoIter;
//
//	//algoIter = m_vRenderAlgorithms.begin();
//
//	//for ( ; algoIter != m_vRenderAlgorithms.end(); algoIter++) {
//	//	(*algoIter)->init();
//	//}
//}

//void
//RenderManager::sendKeyToEngine (char keyCode) {
//	//std::vector<IRenderAlgorithm*>::iterator algoIter;
//
//	//algoIter = m_vRenderAlgorithms.begin();
//
//	//for ( ; algoIter != m_vRenderAlgorithms.end(); algoIter++) {
//	//	(*algoIter)->externCommand (keyCode);
//	//}	
//}