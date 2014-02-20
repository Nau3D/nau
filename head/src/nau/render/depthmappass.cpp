#include <nau/render/depthmappass.h>

#include <nau/geometry/frustum.h>

#include <nau.h>

/*

The lightCam should not exist to the outside world. The pass should expose whaterver params it needs
to control the camera, namely near and far (don't confuse with From and To which relate to the viewing 
camera frustum.

Do not add the lightCam to the RenderManager, and set the light viewing direction based on the lights 
direction.

*/


using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::scene;

DepthMapPass::DepthMapPass (const std::string &passName) :
	Pass (passName)
{
	m_ClassName = "depthmap";
	m_pViewport = new Viewport;
	std::string camName = passName + "-LightCam";
	m_LightCamera = RENDERMANAGER->getCamera(camName);
}

DepthMapPass::~DepthMapPass(void)
{
	delete m_LightCamera;
}

void
DepthMapPass::addLight(const std::string &lightName)
{
	m_Lights.push_back (lightName);

	Light *light = RENDERMANAGER->getLight (m_Lights[0]);

	m_pViewport->setProp(Viewport::SIZE, vec2(m_RenderTarget->getWidth(), m_RenderTarget->getHeight()));
	m_LightCamera->setViewport (m_pViewport);
	

	// common properties to both direction and point lights
	vec4 v = light->getPropf4(Light::DIRECTION);
	m_LightCamera->setProp(Camera::VIEW_VEC, v.x,v.y,v.z,0.0f);

	// although directional lights do not have a position
	v = light->getPropf4(Light::POSITION);
	m_LightCamera->setProp(Camera::POSITION, v.x,v.y,v.z,1.0f);
	//m_LightCamera->setUpVector(0.0f,1.0f,0.0f);

	if (light->getPrope(Light::TYPE) == Light::DIRECTIONAL) {
	
		m_LightCamera->setOrtho(-100.0f, 100.0f,-100.0f, 100.0f,0.0f,200.0f);
		m_LightCamera->setProp(Camera::PROJECTION_TYPE, Camera::ORTHO);
	}
	else {
		m_LightCamera->setPerspective(60.0f, 0.1f, 100.0f);
		m_LightCamera->setProp(Camera::PROJECTION_TYPE, Camera::PERSPECTIVE);
	}
}

void
DepthMapPass::prepare (void)
{
	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->bind();
	}

	prepareBuffers();

	RENDERER->setCamera(m_LightCamera);
}


void
DepthMapPass::restore (void)
{
	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->unbind();
	}
}



//void
//DepthMapPass::setupCamera() {
//
//	RENDERER->setMatrix (IRenderer::PROJECTION);
//	RENDERER->loadIdentity();
//
//	RENDERER->setMatrix (IRenderer::MODELVIEW);
//	RENDERER->loadIdentity();
//
//	RENDERER->setCamera (*m_LightCamera);
//}


void 
DepthMapPass::doPass (void)
{

	// THIS ONLY WORKS WITH ONE DIRECTIONAL LIGHT, MULTIPLE LIGHTS IMPLY MULTIPLE PASSES //
	
	//RENDERMANAGER->getRenderer()->setDepthCompare();
		
	Frustum frustum;
	float cNear, cFar;

	Camera *aCamera = RENDERMANAGER->getCamera(m_CameraName);

	cNear = aCamera->getPropf(Camera::NEARP);
	cFar = aCamera->getPropf(Camera::FARP);

	if (m_Paramf.count("From"))
		cNear = m_Paramf["From"];
	if (m_Paramf.count("To"))
		cFar = m_Paramf["To"];
		
	m_LightCamera->adjustMatrixPlus(cNear,cFar,aCamera);
	Light *l = RENDERMANAGER->getLight(m_Lights[0]);
	vec4 v = m_LightCamera->getPropf4(Camera::VIEW_VEC);
	l->setProp(Light::DIRECTION, v);

	RENDERER->setCamera(m_LightCamera);
	frustum.setFromMatrix (RENDERER->getMatrix(IRenderer::PROJECTION_VIEW_MODEL));

	RENDERMANAGER->clearQueue();

	std::vector<std::string>::iterator scenesIter;
	scenesIter = m_SceneVector.begin();

	for ( ; scenesIter != m_SceneVector.end(); ++scenesIter) {
		IScene *aScene = RENDERMANAGER->getScene (*scenesIter);

		std::vector<SceneObject*> &sceneObjects = aScene->findVisibleSceneObjects (frustum, *m_LightCamera,true);
		//std::vector<SceneObject*> &sceneObjects = aScene->getAllObjects();
		std::vector<SceneObject*>::iterator objIter;

		//m_LightCamera->setNearAndFarPlanes (-aScene->getBoundingVolume().getMax().y, -aScene->getBoundingVolume().getMin().y);
		
		objIter = sceneObjects.begin();
		for (; objIter != sceneObjects.end(); ++objIter) {
			RENDERMANAGER->addToQueue (*objIter, m_MaterialMap);
		}
	}

	RENDERER->setCullFace( IRenderer::FRONT);
//	RENDERER->enableDepthTest();

#if NAU_CORE_OPENGL == 0
	RENDERER->deactivateLighting();
	RENDERER->disableTexturing();
#endif
	RENDERER->setProp(IRenderer::DEPTH_CLAMPING, true);

	RENDERMANAGER->processQueue();

	RENDERER->setProp(IRenderer::DEPTH_CLAMPING, false);

#if NAU_CORE_OPENGL == 0
	RENDERER->enableTexturing();
#endif
	RENDERER->setCullFace (IRenderer::BACK);
	//RENDERER->disableDepthTest();
}

