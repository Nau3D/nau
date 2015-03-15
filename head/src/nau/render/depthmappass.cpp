#include "nau/render/depthmappass.h"

#include "nau/geometry/frustum.h"

#include "nau.h"

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


bool DepthMapPass::Inited = Init();

bool
DepthMapPass::Init() {
	//// FLOAT
	//Attribs.add(Attribute(FROM, "FROM", Enums::DataType::FLOAT, false, new float(-1)));
	//Attribs.add(Attribute(TO, "TO", Enums::DataType::FLOAT, false, new float(-1)));

	return true;
}


DepthMapPass::DepthMapPass (const std::string &passName) :
	Pass (passName)
{
	m_ClassName = "depthmap";
	m_Viewport = new Viewport;
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

	m_Viewport->setPropf2(Viewport::SIZE, vec2(m_RenderTarget->getWidth(), m_RenderTarget->getHeight()));
	m_LightCamera->setViewport (m_Viewport);
	

	// common properties to both direction and point lights
	vec4 v = light->getPropf4(Light::DIRECTION);
	m_LightCamera->setPropf4(Camera::VIEW_VEC, v.x,v.y,v.z,0.0f);

	// although directional lights do not have a position
	v = light->getPropf4(Light::POSITION);
	m_LightCamera->setPropf4(Camera::POSITION, v.x,v.y,v.z,1.0f);
	//m_LightCamera->setUpVector(0.0f,1.0f,0.0f);

	if (light->getPropf(Light::SPOT_CUTOFF) == 180.0f) {
	
		m_LightCamera->setOrtho(-100.0f, 100.0f,-100.0f, 100.0f,0.0f,200.0f);
		m_LightCamera->setPrope(Camera::PROJECTION_TYPE, Camera::ORTHO);
	}
	else {
		m_LightCamera->setPerspective(60.0f, 0.1f, 100.0f);
		m_LightCamera->setPrope(Camera::PROJECTION_TYPE, Camera::PERSPECTIVE);
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
	setupLights();
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

	// THIS ONLY WORKS WITH ONE DIRECTIONAL LIGHT, MULTIPLE LIGHTS REQUIRE MULTIPLE PASSES //
	
	//RENDERMANAGER->getRenderer()->setDepthCompare();
		
	Frustum frustum;
	float cNear, cFar;

	m_LightCamera->setPropf4(Camera::UP_VEC, 0,1,0,0);
	vec4 l = RENDERMANAGER->getLight(m_Lights[0])->getPropf4(Light::DIRECTION);
	m_LightCamera->setPropf4(Camera::VIEW_VEC,l.x,l.y,l.z,l.w);

	Camera *aCamera = RENDERMANAGER->getCamera(m_CameraName);

	cNear = aCamera->getPropf(Camera::NEARP);
	cFar = aCamera->getPropf(Camera::FARP);

	int idFrom = Attribs.getID("FROM");
	int idTo = Attribs.getID("TO");
	if (idFrom != -1)
		cNear = m_FloatProps[idFrom];
	if (idTo != -1)
		cFar = m_FloatProps[idTo];

	//if (m_Paramf.count("From"))
	//	cNear = m_Paramf["From"];
	//if (m_Paramf.count("To"))
	//	cFar = m_Paramf["To"];
		
	m_LightCamera->adjustMatrixPlus(cNear,cFar,aCamera);

	RENDERER->setCamera(m_LightCamera);
	frustum.setFromMatrix ((float *)RENDERER->getProp(IRenderer::PROJECTION_VIEW_MODEL, Enums::MAT4));

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

//	RENDERER->setCullFace( IRenderer::FRONT);
//	RENDERER->enableDepthTest();

	RENDERER->setDepthClamping(true);

	RENDERMANAGER->processQueue();

	RENDERER->setDepthClamping(false);

//	RENDERER->setCullFace (IRenderer::BACK);
	//RENDERER->disableDepthTest();
}

