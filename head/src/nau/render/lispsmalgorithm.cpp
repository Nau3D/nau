#include <render/lispsmalgorithm.h>

#include <materials/imaterialgroup.h>
#include <render/vertexdata.h>
#include <scene/isceneobject.h>
#include <scene/camera.h>
#include <geometry/frustum.h>
#include <math/itransform.h>

#include <math/simpletransform.h>
#include <math/vecbase.h>

#define SHADOWMAPSIZE 2048
#define DISPLAYSIZE 1024

using namespace nau::render;
using namespace nau::scene;
using namespace nau::geometry;
using namespace nau::materials;
using namespace nau::math;

LiSPSMAlgorithm::LiSPSMAlgorithm(void) : m_Inited (false), m_Split (0), m_RenderBoundBox (false), m_FixedFunc (true)
{
}

LiSPSMAlgorithm::~LiSPSMAlgorithm(void)
{
	delete m_ShadowTexture;
	delete m_RenderTarget;

	delete m_pBlankShader;

	delete m_Quad;
}

void
LiSPSMAlgorithm::setRenderer (IRenderer *aRenderer) 
{
	m_pRenderer = aRenderer;
}

void
LiSPSMAlgorithm::init (void)
{

	if (false == m_Inited) {
		m_Inited = true;


		m_CausticTexture = new CTextureDevIL();
		m_CausticTexture->setFilename ("c:/pl3d/texturas/caustic.jpg");

		m_CausticTexture->prepare();
		glTexGeni (GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
		glTexGeni (GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);

		m_CausticTexture->restore();


		//m_ShadowTexture = Texture::create (Texture::TEXTURE_2D, Texture::DEPTH_COMPONENT24, Texture::DEPTH_COMPONENT, Texture::UNSIGNED_BYTE, 1024, 1024);
		m_ShadowTexture = Texture::create (Texture::TEXTURE_2D, Texture::RGBA32F, Texture::RGBA, Texture::FLOAT, SHADOWMAPSIZE, SHADOWMAPSIZE);


		//m_LightCamTexture = Texture::create (Texture::TEXTURE_2D, Texture::RGBA8, Texture::RGBA, Texture::UNSIGNED_BYTE, 2048, 2048);

		//m_DepthTexture = Texture::create (Texture::TEXTURE_2D, Texture::DEPTH_COMPONENT24, Texture::DEPTH_COMPONENT, Texture::UNSIGNED_BYTE, 1024, 1024);

		m_RenderTarget = RenderTarget::create(SHADOWMAPSIZE, SHADOWMAPSIZE);
		
		m_AmbientTexture = Texture::create (Texture::TEXTURE_2D, Texture::RGBA8, Texture::RGBA, Texture::UNSIGNED_BYTE, DISPLAYSIZE, DISPLAYSIZE);
		
		m_NormalTexture = Texture::create (Texture::TEXTURE_2D, Texture::RGBA8, Texture::RGBA, Texture::UNSIGNED_BYTE, DISPLAYSIZE, DISPLAYSIZE);

		m_FinalTexture = Texture::create (Texture::TEXTURE_2D, Texture::RGBA8, Texture::RGBA, Texture::UNSIGNED_BYTE, DISPLAYSIZE, DISPLAYSIZE);

		m_MRT = RenderTarget::create(DISPLAYSIZE, DISPLAYSIZE);

		m_WaterReflectionTexture = Texture::create (Texture::TEXTURE_2D, Texture::RGBA8, Texture::RGBA, Texture::UNSIGNED_BYTE, DISPLAYSIZE, DISPLAYSIZE);
		
		//m_DepthTexture = Texture::create (Texture::TEXTURE_2D, Texture::FLOAT32NV, Texture::LUMINANCE, Texture::FLOAT, 1024, 1024);


		/***MARK***/ //Transform in relative path!
		m_pBlankShader = new CProgram ("D:\\pl3d\\data\\shaders\\depthOnly.vert", 
													"D:\\pl3d\\data\\shaders\\depthOnly.frag");

		//m_pBlankShader = new CProgram ("E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\depthOnly.vert", 
		//											"E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\depthOnly.frag");

		//m_pBlankShader = new CProgram ("C:\\shaders\\depthOnly.vert", 
		//								"C:\\shaders\\depthOnly.frag");
		m_pBlankShader->m_useShader = 1;

		m_pMaterialPassShader = new CProgram ("D:\\pl3d\\data\\shaders\\materialPass.vert", 
													"D:\\pl3d\\data\\shaders\\materialPass.frag");

		//m_pMaterialPassShader = new CProgram ("E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\materialPass.vert", 
		//											"E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\materialPass.frag");

		//m_pMaterialPassShader = new CProgram ("C:\\shaders\\materialPass.vert", 
		//										"C:\\shaders\\materialPass.frag");

		m_pMaterialPassShader->m_useShader = 1;


		m_pWaterShader = new CProgram ("D:\\pl3d\\data\\shaders\\waterpass.vert", 
													"D:\\pl3d\\data\\shaders\\waterpass.frag");

		//m_pWaterShader = new CProgram ("E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\waterpass.vert", 
		//											"E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\waterpass.frag");

		//m_pWaterShader = new CProgram ("C:\\shaders\\waterpass.vert", 
		//								"C:\\shaders\\waterpass.frag");

		m_pWaterShader->m_useShader = 1;


		m_pDeferredShader = new CProgram ("D:\\pl3d\\data\\shaders\\deferred.vert", 
													"D:\\pl3d\\data\\shaders\\deferred.frag");

		//m_pDeferredShader = new CProgram ("E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\deferred.vert", 
		//											"E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\deferred.frag");

		//m_pDeferredShader = new CProgram ("C:\\shaders\\deferred.vert", 
		//									"C:\\shaders\\deferred.frag");

		m_pDeferredShader->m_useShader = 1;

		m_pDeferredShadowShader = new CProgram ("D:\\pl3d\\data\\shaders\\deferredshadow.vert", 
													"D:\\pl3d\\data\\shaders\\deferredshadow.frag");

		//m_pDeferredShadowShader = new CProgram ("E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\deferredshadow.vert", 
		//											"E:\\Projects\\dMV\\Active\\pl3d\\data\\shaders\\deferredshadow.frag");

		//m_pDeferredShadowShader = new CProgram ("C:\\shaders\\deferredshadow.vert", 
		//										"C:\\shaders\\deferredshadow.frag");

		m_pDeferredShadowShader->m_useShader = 1;

		m_Quad = new Quad;

		/***MARK***/ //This is not API independent. Only GLSL has attributes

		VertexData& vd = m_Quad->getRenderable().getVertexData(); /***MARK***/ //This has to be done for all the shaders that need the viewVector
		vd.setAttributeLocationFor (VertexData::CUSTOM_ATTRIBUTE_ARRAY0, m_pDeferredShadowShader->getAttributeLocation("viewVector"));

	} else {
		m_pBlankShader->reload() ;
		m_pDeferredShader->reload();
		m_pDeferredShadowShader->reload();
		m_pMaterialPassShader->reload();
		m_pWaterShader->reload();		

		/***MARK***/ //This is not API independent. Only GLSL has attributes

		VertexData& vd = m_Quad->getRenderable().getVertexData(); /***MARK***/ //This has to be done for all the shaders that need the viewVector
		vd.setAttributeLocationFor (VertexData::CUSTOM_ATTRIBUTE_ARRAY0, m_pDeferredShadowShader->getAttributeLocation("viewVector"));

	}
}

void
LiSPSMAlgorithm::renderScene (IScene *aScene)
{

	Light *sunLight = aScene->getLight("Sun");
	Camera *lightCam = aScene->getCamera ("SunCamera");
	Camera *quadCam = aScene->getCamera ("QuadCamera");

	m_pRenderer->enableDepthTest();
	m_pRenderer->setMaterialProvider (aScene); /***MARK***/ //!!!!!!!!!!!!!

	/* Do we need to calculate shadow?*/

	std::vector<Camera*> &vSceneCameras = aScene->getCameras();
	std::vector<Camera*>::iterator camerasIter;	

	camerasIter = vSceneCameras.begin();

	for ( ; camerasIter != vSceneCameras.end(); camerasIter++) {
		Camera *aCamera = (*camerasIter);

		if (false == aCamera->isActive()) {
			continue;
		}

		/*Calculate Shadow*/
		if (false == m_FixedFunc) {
			calculateShadow (aScene, *aCamera);	
		}

		m_pRenderer->setMatrix (nau::render::IRenderer::PROJECTION);
		m_pRenderer->loadIdentity();

		m_pRenderer->setMatrix (nau::render::IRenderer::MODELVIEW);
		m_pRenderer->loadIdentity();	

		m_pRenderer->setCamera (*aCamera);

		m_pRenderer->activateLight (*sunLight);
		m_pRenderer->positionLight (*sunLight);

		m_pRenderer->unproject (m_Quad->getRenderable(), *aCamera);

		Frustum frustum;

		frustum.setFromMatrix (m_pRenderer->getProjectionModelviewMatrix());

		std::vector<ISceneObject*> &sceneObjects = aScene->findVisibleSceneObjects (frustum, *aCamera);

		if (true == m_FixedFunc) {
			m_pRenderer->clear (IRenderer::COLORBUFFER | IRenderer::DEPTHBUFFER);
			if (true == m_pRenderer->isStereo()) {
				aCamera->setPositionOffset (0.035);

				m_pRenderer->setMatrix (nau::render::IRenderer::PROJECTION);
				m_pRenderer->push();
				m_pRenderer->loadIdentity();

				m_pRenderer->setMatrix (nau::render::IRenderer::MODELVIEW);
				m_pRenderer->push();
				m_pRenderer->loadIdentity();	

				m_pRenderer->setCamera (*aCamera);
				m_pRenderer->colorMask (0, 1, 1, 1);
				renderFixed (sceneObjects, *aCamera);

				m_pRenderer->setMatrix (nau::render::IRenderer::PROJECTION);
				m_pRenderer->pop();

				m_pRenderer->setMatrix (nau::render::IRenderer::MODELVIEW);
				m_pRenderer->pop();	

				aCamera->setPositionOffset (-0.035);

				m_pRenderer->setMatrix (nau::render::IRenderer::PROJECTION);
				m_pRenderer->push();
				m_pRenderer->loadIdentity();

				m_pRenderer->setMatrix (nau::render::IRenderer::MODELVIEW);
				m_pRenderer->push();
				m_pRenderer->loadIdentity();	

				m_pRenderer->setCamera (*aCamera);

				m_pRenderer->colorMask (1, 0, 0, 0);
				m_pRenderer->clear (IRenderer::DEPTHBUFFER);
			}
			renderFixed (sceneObjects, *aCamera);
			if (true == m_pRenderer->isStereo()) {
				m_pRenderer->setMatrix (nau::render::IRenderer::PROJECTION);
				m_pRenderer->pop();

				m_pRenderer->setMatrix (nau::render::IRenderer::MODELVIEW);
				m_pRenderer->pop();	

				m_pRenderer->colorMask (1, 1, 1, 1);
				aCamera->setPositionOffset (0.0f);
			}
			continue;
		}
	
		
		// Shadow matrix
		SimpleTransform t; /***MARK***/ //Hide this behind a factory?

		t.translate (0.5f, 0.5f, 0.5f);	
		t.scale (0.5f);

		t.compose (lightCam->getProjectionMatrix());
		t.compose (lightCam->getViewMatrix());
		
		t.compose (aCamera->getViewMatrixInverse());

/*		m_pRenderer->setMatrix (nau::render::IRenderer::PROJECTION);
		m_pRenderer->loadIdentity();

		m_pRenderer->setMatrix (nau::render::IRenderer::MODELVIEW);
		m_pRenderer->loadIdentity();	*/	

		//m_pRenderer->setCamera (*aCamera);

		//m_pRenderer->activateLight (*sunLight);
		//m_pRenderer->positionLight (*sunLight);

		//m_pRenderer->unproject (m_Quad->getRenderable(), *aCamera);

		////m_pRenderer->clear();

		//Frustum frustum;

		//frustum.setFromMatrix (m_pRenderer->getProjectionModelviewMatrix());

		//std::vector<ISceneObject*> &sceneObjects = aScene->findVisibleSceneObjects (frustum, *aCamera);


		/*Do we have water*/
		float plane[4];
		bool water = waterOnFrustum (aScene, sceneObjects, plane);

		if (true == water) {

			m_pRenderer->setMatrix (IRenderer::MODELVIEW);
			m_pRenderer->push();
			

			m_pRenderer->translate (vec3 (0.0f, 8.0f, 0.0f)); /***MARK***/ //Water level!
	  	    m_pRenderer->scale (vec3 (1.0f, -1.0f, 1.0f));
	
			m_pRenderer->activateUserClipPlane (IRenderer::CLIP_PLANE1);
			m_pRenderer->setUserClipPlane (IRenderer::CLIP_PLANE1, (double *)plane);

			m_WaterMaterial->m_enabled = false; /***MARK***/ //This should be materialmanager.disable ("water");

			m_pRenderer->setCullFace (IRenderer::FRONT);

			materialPass (sceneObjects, t);

			m_pRenderer->setCullFace (IRenderer::BACK);

			m_pRenderer->deactivateUserClipPlane (IRenderer::CLIP_PLANE1);

			m_MRT->bind();
			m_MRT->attachColorTexture (m_WaterReflectionTexture, RenderTarget::COLOR0);
			m_MRT->setDrawBuffers();
			
			//m_MRT->attachDepthTexture (m_DepthTexture);
			
			deferredShadePass (*quadCam, *aCamera, *sunLight);

			m_MRT->dettachColorTexture (RenderTarget::COLOR0);
			//m_MRT->dettachDepthTexture();
			m_MRT->unbind();

			m_WaterMaterial->m_enabled = true;

			m_pRenderer->setMatrix (IRenderer::MODELVIEW);
			m_pRenderer->pop();
		}


		materialPass (sceneObjects, t);

		m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

		if (true == water) {
			aScene->setAllMaterialsTo (false);
			waterPass (sceneObjects);
			aScene->setAllMaterialsTo (true);
		}

		//m_MRT->bind();
		//m_MRT->attachColorTexture (m_FinalTexture, RenderTarget::COLOR0);
		//m_MRT->setDrawBuffers();

		deferredShadePass (*quadCam, *aCamera, *sunLight);

		//m_MRT->dettachColorTexture (RenderTarget::COLOR0);
		//m_MRT->unbind();

		//drawShadow(*quadCam, *lightCam, *aCamera);

		m_pRenderer->deactivateLight (*sunLight);
	}

	m_pRenderer->disableDepthTest();
	//debug(*quadCam);	
}


void
LiSPSMAlgorithm::renderFixed (std::vector<ISceneObject*> &sceneObjects, Camera &aCam)
{
	m_pRenderer->activateLighting();
	m_pRenderer->enableSurfaceShaders();
	m_pRenderer->enableDepthTest();

	m_pRenderer->startRender();

	std::vector<ISceneObject*>::iterator objIter;
	
	objIter = sceneObjects.begin();
	
	bool fog = false;
	float planeHeight = 0.0f;
	ISceneObject* waterObject = 0;
	
	for ( ; objIter != sceneObjects.end(); objIter++) {
		
		ISceneObject *aObject = (*objIter);

		if (0 == aObject->getName().compare ("pPlane1")) {
			planeHeight = aObject->_getTransformPtr()->getTranslation().y + 4.227f;
			if (aCam.getPosition().y  < planeHeight) { //ATTENTION TO THE MAGIC 3.8f
				waterObject = aObject;
				fog = true;
			}
		}
		if (true == m_RenderBoundBox) {
			const ITransform& aTransform = aObject->getTransform();

			//m_pRenderer->push();
			//m_pRenderer->applyTransform (aTransform);
			m_pRenderer->renderBoundingVolume (aObject->getBoundingVolume());
			//m_pRenderer->pop();
		}		
		m_pRenderer->renderObject (*aObject);


	}
	//}

	if (true == fog) {
		double plane[4] = {0.0f, -1.0f, 0.0f, planeHeight };

		m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT1);

		m_CausticTexture->prepare();

		glTexGeni (GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR); /***MARK***/ //OpenGL calls!
		glTexGeni (GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);

		
		float planeS[] = {-0.1f, -0.1f, 0.0f, 0.0f};
		float planeT[] = {0.0f, 0.0f, 0.14f, 0.0f};

		glTexGenfv (GL_S, GL_OBJECT_PLANE, planeS);
		glTexGenfv (GL_T, GL_OBJECT_PLANE, planeT);
		
		m_pRenderer->setMatrix (IRenderer::TEXTURE);
		m_pRenderer->loadIdentity();
		
		float timeAgo = DegToRad (clock() * 2.0f / CLOCKS_PER_SEC);

		glTranslatef (sin (timeAgo), cos (timeAgo), 0.0f);

		m_pRenderer->setMatrix (IRenderer::MODELVIEW);

		m_pRenderer->enableTextureCoordsGen();
		
		//m_pRenderer->setMatrix (IRenderer::TEXTURE);
		//m_pRenderer->loadIdentity();
		//m_pRenderer->applyTransform (aCam.getViewMatrixInverse());
		//m_pRenderer->setMatrix (IRenderer::MODELVIEW);


		m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);
		m_pRenderer->enableFog();
		////m_pRenderer->push();
		////m_pRenderer->loadIdentity();
		m_pRenderer->activateUserClipPlane(IRenderer::CLIP_PLANE0);
		m_pRenderer->setUserClipPlane (IRenderer::CLIP_PLANE0, plane);
		////m_pRenderer->pop();

		
		m_pRenderer->finishRender();
		
		m_pRenderer->disableFog();

		m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT1);
		m_pRenderer->disableTextureCoordsGen();
		m_CausticTexture->restore();

		m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);
		plane[1] = 1.0f;
		plane[3] = -planeHeight;
		m_pRenderer->setUserClipPlane (IRenderer::CLIP_PLANE0, plane);
		m_pRenderer->finishRender();

		m_pRenderer->deactivateUserClipPlane (IRenderer::CLIP_PLANE0);

		//m_pRenderer->clear (IRenderer::DEPTHBUFFER);
		m_pRenderer->setCullFace (IRenderer::FRONT);
		m_pRenderer->startRender();
		m_pRenderer->renderObject (*waterObject);
		m_pRenderer->finishRender();
		m_pRenderer->setCullFace (IRenderer::BACK);

		
	} else {

		m_pRenderer->finishRender();
	}

	m_pRenderer->disableSurfaceShaders();
	m_pRenderer->disableDepthTest();
	m_pRenderer->deactivateLighting();
}

void
LiSPSMAlgorithm::calculateShadow (IScene *aScene, Camera &aCamera)
{
	float splits[5] = { 1.5f, 5.0f, 15.0f, 50.0f, 15000.0f };

	Light *sunLight = aScene->getLight("Sun");
	Camera *lightCam = aScene->getCamera ("SunCamera");
	
	nau::render::Viewport& lightViewport = lightCam->getViewport();
	lightViewport.setSize (SHADOWMAPSIZE, SHADOWMAPSIZE);
	lightViewport.setBackgroundColor (vec4 (1.0f, 1.0f, 1.0f, 1.0f));

	lightCam->setCamera (sunLight->getPosition(), sunLight->getDirection(), vec3 (0.0f, 1.0f, 0.0f));

	/* SHADOW CALCULATION PASS
	- set light camera 
	- activate render target (v)
	- set blank shader (v)
	- calculate frustum
	- setup camera (v)
	- render
	- set shadow shader
	- for each camera: render
	*/

	//m_ShadowTexture->bind();
	//m_ShadowTexture->enableCompareToTexture();
	//m_ShadowTexture->unbind();

	m_RenderTarget->bind();
	m_RenderTarget->attachColorTexture (m_ShadowTexture, RenderTarget::COLOR0);
	m_RenderTarget->setDrawBuffers();


//	std::vector<ISceneObject*> &sceneObjects = aScene->getAllObjects();
	std::vector<ISceneObject*>::iterator objIter;

	m_pBlankShader->useProgram();

	for (int i = 0; i < 4; i++) { /***MARK**/

		m_pRenderer->enableDepthTest();
		lightCam->adjustMatrix (splits[i], splits[i+1], aCamera);

		m_pRenderer->setMatrix (IRenderer::PROJECTION);
		m_pRenderer->push();
		m_pRenderer->loadIdentity();
		m_pRenderer->setMatrix (IRenderer::MODELVIEW);
		m_pRenderer->push();
		m_pRenderer->loadIdentity();

		m_pRenderer->setCamera (*lightCam);

		const_cast<mat4&>(m_LightTransforms[i].getMat44()).copy (lightCam->getModelViewProjection().getMat44());

		Frustum frustum;

		frustum.setFromMatrix (m_pRenderer->getProjectionModelviewMatrix());
		std::vector<ISceneObject*> &sceneObjects = aScene->findVisibleSceneObjects (frustum, aCamera);

		objIter = sceneObjects.begin();

		if (0 == i) {
			m_pRenderer->clear(IRenderer::COLORBUFFER | IRenderer::DEPTHBUFFER);
		} else {
			m_pRenderer->clear(IRenderer::DEPTHBUFFER);
		}

		if (0 == i) {
			m_pRenderer->colorMask (1, 0, 0, 0);
		} else if (1 == i) {
			m_pRenderer->colorMask (0, 1, 0, 0);
		} else if (2 == i) {
			m_pRenderer->colorMask (0, 0, 1, 0);
		} else if (3 == i) {
			m_pRenderer->colorMask (0, 0, 0, 1);
		}

		m_pRenderer->setCullFace (IRenderer::FRONT);

		m_pRenderer->startRender();


		for ( ; objIter != sceneObjects.end(); objIter++) {
			ISceneObject *aObject = (*objIter);
			const ITransform& aTransform = aObject->getTransform();

			m_pRenderer->push();
			m_pRenderer->applyTransform (aTransform);
			//m_pRenderer->renderBoundingVolume (aObject->getBoundingVolume());
			m_pRenderer->renderObject (*aObject, VertexData::DRAW_VERTICES);
			m_pRenderer->pop();
		}

		m_pRenderer->finishRender();

		m_pRenderer->setCullFace (IRenderer::BACK);

		m_pRenderer->setMatrix (IRenderer::PROJECTION);
		m_pRenderer->pop();
		m_pRenderer->setMatrix (IRenderer::MODELVIEW);
		m_pRenderer->pop();

		m_pRenderer->disableDepthTest();
	}

	m_pRenderer->colorMask (1, 1, 1, 1);

	m_RenderTarget->dettachColorTexture (RenderTarget::COLOR0);
	//m_RenderTarget->dettachDepthTexture();
    m_RenderTarget->unbind();

	CProgram::FixedFunction();

	/**Draw a light*/

/*
	lightCam->adjustMatrix (splits[m_Split], splits[m_Split + 1], aCamera);

	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();
	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	lightViewport.setSize (1024, 1024);
	lightViewport.setBackgroundColor (vec4 (0.0f, 0.0f, 0.0f, 1.0f));

	//m_pRenderer->setCamera (aCamera);
	m_pRenderer->setCamera (*lightCam);

	Frustum frustum;

	frustum.setFromMatrix (m_pRenderer->getProjectionModelviewMatrix());
	std::vector<ISceneObject*> & sceneObjects = aScene->findVisibleSceneObjects (frustum, aCamera);

	objIter = sceneObjects.begin();

	m_pRenderer->clear(IRenderer::COLORBUFFER | IRenderer::DEPTHBUFFER);
	//m_pRenderer->setColor (1.0f, 1.0f, 1.0f, 1.0f);

	//m_pRenderer->activateLighting();
	m_pRenderer->activateLight (*sunLight);
	m_pRenderer->positionLight (*sunLight);

	objIter = sceneObjects.begin();

	m_pRenderer->setCullFace (IRenderer::FRONT);

	m_pRenderer->startRender();
	m_pRenderer->enableSurfaceShaders();

	for ( ; objIter != sceneObjects.end(); objIter++) {
		ISceneObject *aObject = (*objIter);
		const ITransform& aTransform = aObject->getTransform();

		m_pRenderer->push();
		m_pRenderer->applyTransform (aTransform);
		//m_pRenderer->renderBoundingVolume (aObject->getBoundingVolume());
		m_pRenderer->renderObject (aObject->getRenderable());
		m_pRenderer->pop();
	}

	m_pRenderer->finishRender();
	m_pRenderer->setCullFace (IRenderer::BACK);
	m_pRenderer->disableSurfaceShaders();

	m_pRenderer->deactivateLight (*sunLight);
	m_pRenderer->deactivateLighting();


	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->pop();
	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->pop();
	*/
}

void
LiSPSMAlgorithm::materialPass (std::vector<ISceneObject*> &sceneObjects, ITransform &t)
{
	m_pRenderer->enableSurfaceShaders();
	m_pRenderer->enableDepthTest();

	//m_DepthTexture->bind();
	//m_DepthTexture->enableCompareToTexture();
	//m_DepthTexture->unbind();


	m_MRT->bind();
	m_MRT->attachColorTexture (m_AmbientTexture, RenderTarget::COLOR0);
	m_MRT->attachColorTexture (m_NormalTexture, RenderTarget::COLOR1);
	m_MRT->setDrawBuffers();
	

	m_pMaterialPassShader->useProgram();
	m_pMaterialPassShader->setValueOfUniformByNamef ("albedoTexture", 0);
	m_pMaterialPassShader->setValueOfUniformByNamef ("normalTexture", 1);
	m_pMaterialPassShader->setValueOfUniformByNamef ("depthTexture", 2);

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT2);
	m_ShadowTexture->bind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

	m_pRenderer->clear (IRenderer::COLORBUFFER | IRenderer::DEPTHBUFFER);

	/* Set light matrices */

	for (int i = 0; i < 4; i++) {
		m_pRenderer->setActiveTextureUnit ((IRenderer::TextureUnit)(IRenderer::TEXTURE_UNIT0 + i));
		m_pRenderer->setMatrix (IRenderer::TEXTURE);

		m_pRenderer->loadIdentity();
		m_pRenderer->translate (vec3 (0.5f, 0.5f, 0.5f));
		m_pRenderer->scale (vec3 (0.5f, 0.5f, 0.5f));

		m_pRenderer->applyTransform (m_LightTransforms[i]);
	}

	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

	m_pRenderer->startRender();

	m_pRenderer->setShader (m_pMaterialPassShader);

	std::vector<ISceneObject*>::iterator objIter;
	
	objIter = sceneObjects.begin();

	for ( ; objIter != sceneObjects.end(); objIter++) {
		ISceneObject *aObject = (*objIter);
		const ITransform& aTransform = aObject->getTransform();

		//m_pRenderer->push();
		//m_pRenderer->applyTransform (aTransform);
		if (true == m_RenderBoundBox) {
			m_pRenderer->renderBoundingVolume (aObject->getBoundingVolume());
		}

		VertexData& vd = aObject->getRenderable().getVertexData();
		vd.setAttributeLocationFor (VertexData::CUSTOM_ATTRIBUTE_ARRAY1, m_pMaterialPassShader->getAttributeLocation("tgt"));
		vd.setAttributeLocationFor (VertexData::CUSTOM_ATTRIBUTE_ARRAY2, m_pMaterialPassShader->getAttributeLocation("bin"));

		
		//m_pRenderer->renderObject (aObject->getRenderable());
		m_pRenderer->renderObject (*aObject);
		//m_pRenderer->pop();
	 }
	//}
	m_pRenderer->finishRender(); /***THERE IS A PROBLEM WITH THIS FUNCTION!!! The objects are not being multiplied by MODELVIEW*/

	/* Set light matrices */

	for (int i = 0; i < 4; i++) {
		m_pRenderer->setActiveTextureUnit ((IRenderer::TextureUnit)(IRenderer::TEXTURE_UNIT0 + i));
		m_pRenderer->setMatrix (IRenderer::TEXTURE);

		m_pRenderer->loadIdentity();
	}

	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

	m_pRenderer->setShader (0);

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT2);
	m_ShadowTexture->unbind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

	//m_MRT->dettachColorTexture (RenderTarget::COLOR2);
	m_MRT->dettachColorTexture (RenderTarget::COLOR1);
	m_MRT->dettachColorTexture (RenderTarget::COLOR0);
	//m_MRT->dettachDepthTexture();
	m_MRT->unbind();
	

	m_pRenderer->disableSurfaceShaders();

	CProgram::FixedFunction();
}

void
LiSPSMAlgorithm::deferredShadePass (Camera &quadCam, Camera &aCamera, Light &aLight)
{
	m_pRenderer->clear (IRenderer::COLORBUFFER | IRenderer::DEPTHBUFFER);
	
	m_pRenderer->disableSurfaceShaders();

	m_pRenderer->activateLighting();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);
	m_AmbientTexture->bind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT1);
	m_NormalTexture->bind();

	//m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT2);
	//m_DepthTexture->bind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

	m_pDeferredShader->useProgram();	
	m_pDeferredShader->setValueOfUniformByNamef ("albedoTexture", 0);
	m_pDeferredShader->setValueOfUniformByNamef ("normalTexture", 1);
	//m_pDeferredShader->setValueOfUniformByNamef ("depthTexture", 2);

	m_pRenderer->activateLight (aLight);

	float planes[2];

	planes[0] = - aCamera.getFarPlane() / (aCamera.getFarPlane() - aCamera.getNearPlane());
	planes[1] = - aCamera.getFarPlane() * aCamera.getNearPlane() / (aCamera.getFarPlane() - aCamera.getNearPlane());

	m_pDeferredShader->setValueOfUniformByNamev ("planes", planes);

	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();
	
	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setCamera (quadCam);

	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();
	
	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setCamera (aCamera);
	m_pRenderer->positionLight (aLight);

	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->pop();
	
	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->pop();

	m_pRenderer->startRender();

	m_pRenderer->renderObject (*m_Quad);

	m_pRenderer->finishRender();

	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->pop();
	
	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->pop();
	
	m_pRenderer->deactivateLight (aLight);

	m_pRenderer->deactivateLighting();

	CProgram::FixedFunction();
	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);
	m_AmbientTexture->unbind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT1);
	m_NormalTexture->unbind();

	//m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT2);
	//m_DepthTexture->unbind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);
}



void
LiSPSMAlgorithm::waterPass (std::vector<ISceneObject*> &sceneObjects)
{
	static float startRad  = 0.01f;

	SimpleTransform t;

	t.translate (vec3 (0.5f, 0.5f, 0.5f));	
	t.scale (0.5f);

	m_WaterMaterial->m_enabled = true;

	m_pRenderer->enableSurfaceShaders();

	m_MRT->bind();
	m_MRT->attachColorTexture (m_AmbientTexture, RenderTarget::COLOR0);
	m_MRT->attachColorTexture (m_NormalTexture, RenderTarget::COLOR1);
	//m_MRT->attachDepthTexture (m_DepthTexture);

	m_MRT->setDrawBuffers();
	
	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT2);
	m_WaterReflectionTexture->bind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

	float amplitude[2] = { 0.03f, 0.2f };
	float freq[2] = { 0.02f, 0.03f };

	startRad += 1.0f;

	/*
	if (startRad > 10.0f) {
		startRad = 0.1f;
	}
*/
	m_pWaterShader->useProgram();
	m_pWaterShader->setValueOfUniformByNamef ("albedoTexture", 0);
	m_pWaterShader->setValueOfUniformByNamef ("normalTexture", 1);
	m_pWaterShader->setValueOfUniformByNamef ("reflectionTexture", 2);
	m_pWaterShader->setValueOfUniformByNamef ("StartRad", startRad);


	m_pWaterShader->setValueOfUniformByNamev ("textureMatrix", const_cast<float *> (t.getMat44().getMatrix()));
	m_pWaterShader->setValueOfUniformByNamev ("Amplitude", amplitude);
	m_pWaterShader->setValueOfUniformByNamev ("Freq", freq);

	//m_pRenderer->setCamera (aCamera);

	//m_pRenderer->clear();

	m_pRenderer->colorMask (1, 1, 1, 0);

	m_pRenderer->startRender();

	m_pRenderer->setShader (m_pWaterShader);

	//m_pRenderer->disableDepthTest();

	std::vector<ISceneObject*>::iterator objIter;
	
	objIter = sceneObjects.begin();

	for ( ; objIter != sceneObjects.end(); objIter++) {
		ISceneObject *aObject = (*objIter);
		const ITransform& aTransform = aObject->getTransform();

		m_pRenderer->push();
		m_pRenderer->applyTransform (aTransform);
		//m_pRenderer->renderBoundingVolume (aObject->getBoundingVolume());

		VertexData& vd = aObject->getRenderable().getVertexData();
		vd.setAttributeLocationFor (VertexData::CUSTOM_ATTRIBUTE_ARRAY1, m_pWaterShader->getAttributeLocation("tgt"));
		vd.setAttributeLocationFor (VertexData::CUSTOM_ATTRIBUTE_ARRAY2, m_pWaterShader->getAttributeLocation("bin"));

		
		m_pRenderer->renderObject (*aObject);
		m_pRenderer->pop();
	 }
	//}
	m_pRenderer->finishRender();

	m_pRenderer->colorMask (1, 1, 1, 1);

	//m_pRenderer->enableDepthTest();
	m_pRenderer->setShader (0);

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT2);
	m_WaterReflectionTexture->unbind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

	m_MRT->dettachColorTexture (RenderTarget::COLOR1);
	m_MRT->dettachColorTexture (RenderTarget::COLOR0);
	//m_MRT->dettachDepthTexture();

	m_MRT->unbind();

	m_pRenderer->disableSurfaceShaders();

	CProgram::FixedFunction();

}


void
LiSPSMAlgorithm::drawShadow (Camera &quadCam, Camera &lightCam, Camera &aCamera)
{
	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->loadIdentity();
	m_pRenderer->setMatrix (nau::render::IRenderer::MODELVIEW);
	m_pRenderer->loadIdentity();
	m_pRenderer->setCamera (quadCam);
	m_pRenderer->clear (IRenderer::COLORBUFFER | IRenderer::DEPTHBUFFER);
	
	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);
	m_FinalTexture->bind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT1);
	m_PositionTexture->bind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT2);
	m_ShadowTexture->bind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

	m_pDeferredShadowShader->useProgram();	
	
	m_pDeferredShadowShader->setValueOfUniformByNamef ("renderTexture", 0);
	m_pDeferredShadowShader->setValueOfUniformByNamef ("depthTexture", 1);
	m_pDeferredShadowShader->setValueOfUniformByNamef ("shadowTexture", 2);
	
	// Shadow matrix
	SimpleTransform t; /***MARK***/ //Hide this behind a factory?

	t.translate (0.5f, 0.5f, 0.5f);	
	t.scale (0.5f);

	t.compose (lightCam.getProjectionMatrix());
	t.compose (lightCam.getViewMatrix());
	
	t.compose (aCamera.getViewMatrixInverse());

	//float planes[2];

	//planes[0] = - aCamera.getFarPlane() / (aCamera.getFarPlane() - aCamera.getNearPlane());
	//planes[1] = - aCamera.getFarPlane() * aCamera.getNearPlane() / (aCamera.getFarPlane() - aCamera.getNearPlane());

	//m_pDeferredShadowShader->setValueOfUniformByNamev ("planes", planes);
	//m_pDeferredShadowShader->setValueOfUniformByNamev ("shadowMatrix", const_cast<float*> (t.getMat44().getMatrix()));

	m_pRenderer->startRender();

	m_pRenderer->renderObject (*m_Quad);

	m_pRenderer->finishRender();

	CProgram::FixedFunction();
	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);
	m_FinalTexture->unbind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT1);
	m_PositionTexture->unbind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT2);
	m_ShadowTexture->unbind();

	m_pRenderer->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);

}

void
LiSPSMAlgorithm::debug (Camera &quadCam)
{

	nau::render::Viewport& quadViewport = quadCam.getViewport();

	vec3 qOrg = quadViewport.getOrigin();
	vec3 qSize = quadViewport.getSize();


	m_pRenderer->disableSurfaceShaders();

	m_pRenderer->startRender();

	/*Albedo*/
	quadViewport.setSize (200, 200);
	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setCamera (quadCam);

	m_pRenderer->disableDepthTest();
	m_pRenderer->enableTexturing();
	m_AmbientTexture->bind();
	m_pRenderer->renderObject (*m_Quad);
	m_AmbientTexture->unbind();
	m_pRenderer->disableTexturing();
	m_pRenderer->enableDepthTest();
	
	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->pop();

	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->pop();

	/*Normal*/
	//quadViewport.setOrigin (200, 0);
	//quadViewport.setSize (200, 200);
	//m_pRenderer->setMatrix (IRenderer::PROJECTION);
	//m_pRenderer->push();
	//m_pRenderer->loadIdentity();

	//m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	//m_pRenderer->push();
	//m_pRenderer->loadIdentity();

	//m_pRenderer->setCamera (quadCam);
	//
	//m_pRenderer->disableDepthTest();
	//m_pRenderer->enableTexturing();
	//m_NormalTexture->bind();
	//m_pRenderer->renderObject (m_Quad->getRenderable());
	//m_NormalTexture->unbind();
	//m_pRenderer->disableTexturing();
	//m_pRenderer->enableDepthTest();

	//m_pRenderer->setMatrix (IRenderer::PROJECTION);
	//m_pRenderer->pop();

	//m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	//m_pRenderer->pop();


	/*Water reflection*/
	//quadViewport.setOrigin (400, 0);
	//quadViewport.setSize (200, 200);

	//m_pRenderer->setMatrix (IRenderer::PROJECTION);
	//m_pRenderer->push();
	//m_pRenderer->loadIdentity();

	//m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	//m_pRenderer->push();
	//m_pRenderer->loadIdentity();

	//m_pRenderer->setCamera (quadCam);
	//
	//m_pRenderer->disableDepthTest();
	//m_pRenderer->enableTexturing();
	//m_WaterReflectionTexture->bind();
	//m_pRenderer->renderObject (m_Quad->getRenderable());
	//m_WaterReflectionTexture->unbind();
	//m_pRenderer->disableTexturing();
	//m_pRenderer->enableDepthTest();

	//m_pRenderer->setMatrix (IRenderer::PROJECTION);
	//m_pRenderer->pop();

	//m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	//m_pRenderer->pop();
	

	/*Final*/
	//quadViewport.setSize (200, 200);
	//m_pRenderer->setMatrix (IRenderer::PROJECTION);
	//m_pRenderer->push();
	//m_pRenderer->loadIdentity();

	//m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	//m_pRenderer->push();
	//m_pRenderer->loadIdentity();

	//m_pRenderer->setCamera (quadCam);

	//m_pRenderer->disableDepthTest();
	//m_pRenderer->enableTexturing();
	//m_FinalTexture->bind();
	//m_pRenderer->renderObject (m_Quad->getRenderable());
	//m_FinalTexture->unbind();
	//m_pRenderer->disableTexturing();
	//m_pRenderer->enableDepthTest();
	//
	//m_pRenderer->setMatrix (IRenderer::PROJECTION);
	//m_pRenderer->pop();

	//m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	//m_pRenderer->pop();

	/*Shadow 1*/
	quadViewport.setOrigin (200, 0);
	quadViewport.setSize (200, 200);

	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setCamera (quadCam);

	//m_pRenderer->clear();

	m_pRenderer->disableDepthTest();
	//m_pRenderer->setColor (1.0f, 1.0f, 1.0f, 1.0f);
	m_pRenderer->renderObject (*m_Quad);

	m_pRenderer->colorMask (1, 0, 0, 0);
	m_pRenderer->enableTexturing();
	m_ShadowTexture->bind();
	//m_ShadowTexture->disableCompareToTexture();
	m_pRenderer->renderObject (*m_Quad);
	m_ShadowTexture->unbind();
	m_pRenderer->disableTexturing();
	m_pRenderer->colorMask (1, 1, 1, 1);
	m_pRenderer->enableDepthTest();
	
	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->pop();
	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->pop();


	/*Shadow 2*/
	quadViewport.setOrigin (400, 0);
	quadViewport.setSize (200, 200);
	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setCamera (quadCam);

	m_pRenderer->disableDepthTest();
	//m_pRenderer->setColor (1.0f, 1.0f, 1.0f, 1.0f);
	m_pRenderer->renderObject (*m_Quad);

	m_pRenderer->colorMask (0, 1, 0, 0);
	m_pRenderer->enableTexturing();
	m_ShadowTexture->bind();
	//m_ShadowTexture->disableCompareToTexture();
	m_pRenderer->renderObject (*m_Quad);
	m_ShadowTexture->unbind();
	m_pRenderer->disableTexturing();
	m_pRenderer->colorMask (1, 1, 1, 1);
	m_pRenderer->enableDepthTest();
	
	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->pop();
	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->pop();


	/*Shadow 3*/
	quadViewport.setOrigin (600, 0);
	quadViewport.setSize (200, 200);
	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->push();
	m_pRenderer->loadIdentity();

	m_pRenderer->setCamera (quadCam);

	m_pRenderer->disableDepthTest();
	//m_pRenderer->setColor (1.0f, 1.0f, 1.0f, 1.0f);
	m_pRenderer->renderObject (*m_Quad);

	m_pRenderer->colorMask (0, 0, 1, 0);
	m_pRenderer->enableTexturing();
	m_ShadowTexture->bind();
	//m_ShadowTexture->disableCompareToTexture();
	m_pRenderer->renderObject (*m_Quad);
	m_ShadowTexture->unbind();
	m_pRenderer->disableTexturing();
	m_pRenderer->colorMask (1, 1, 1, 1);
	m_pRenderer->enableDepthTest();
	
	m_pRenderer->setMatrix (IRenderer::PROJECTION);
	m_pRenderer->pop();
	m_pRenderer->setMatrix (IRenderer::MODELVIEW);
	m_pRenderer->pop();

	m_pRenderer->finishRender();
	
	quadViewport.setOrigin (0, 0);
	quadViewport.setSize (qSize.x, qSize.y);

	m_pRenderer->enableSurfaceShaders();

}

bool
LiSPSMAlgorithm::waterOnFrustum (IScene *aScene, std::vector<ISceneObject*> &sceneObjects, float *plane)
{
	std::vector<ISceneObject*>::iterator objIter;
	
	objIter = sceneObjects.begin();

	for ( ; objIter != sceneObjects.end(); objIter++) {
		ISceneObject *aObject = (*objIter);	
		
		IRenderable &renderable = aObject->getRenderable();
		std::vector<IMaterialGroup*> &matGroups = renderable.getMaterialGroups();
		std::vector<IMaterialGroup*>::iterator mgIter;

		mgIter = matGroups.begin();

		for ( ; mgIter != matGroups.end(); mgIter++) {
			IMaterialGroup *mg = (*mgIter);
			
			CMaterial *mat = aScene->getMaterialForRender (mg->getMaterialId());

			if (1 == mat->m_shaderid) {
				plane[0] = 0.0f;
				plane[1] = 1.0f;
				plane[2] = 0.0f;
				plane[3] = 4.0f; /****MARK***/ //This is an hack. Got this info from the model

				m_WaterMaterial = mat;

				return true;
			}
		}
	}

	return false;
}

void
LiSPSMAlgorithm::externCommand (char keyCode) {
	if ('4' >= keyCode && '1' <= keyCode) {
		m_Split = keyCode - '0' - 1;
	}

	if ('B' == keyCode) {
		m_RenderBoundBox = !m_RenderBoundBox;
	}

	if ('P' == keyCode) {
		m_FixedFunc = !m_FixedFunc;
		m_pRenderer->setFixed (m_FixedFunc);
	}
}
