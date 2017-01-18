#include "nau/render/passProfiler.h"

#include "nau/debug/profile.h"
#include "nau/geometry/mesh.h"
#include "nau/render/passFactory.h"
#include "nau/resource/fontManager.h"
#include "nau/scene/sceneObject.h"


using namespace nau::resource;

#include "nau.h"



#ifdef _DEBUG
   #ifndef DBG_NEW
      #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
      #define new DBG_NEW
   #endif
#endif  // _DEBUG


using namespace nau::render;

bool PassProfiler::Inited = PassProfiler::Init();


bool
PassProfiler::Init() {

	PASSFACTORY->registerClass("profiler", Create);
	return true;
}


PassProfiler::PassProfiler (const std::string &name) :
	Pass (name)
{
	m_ClassName = "profiler";

	m_CameraName = "__ProfilerCamera";
	m_pCam = RENDERMANAGER->getCamera("__ProfilerCamera");
	m_pSO = SceneObjectFactory::Create("SimpleObject");
	nau::resource::ResourceManager *rm = RESOURCEMANAGER;
	m_pSO->setRenderable(rm->createRenderable("Mesh", rm->makeMeshName("__ProfilerResult", "Profiler")));

	m_Viewport = RENDERMANAGER->createViewport("__Profiler", vec4(0.0f, 0.0f, 0.0f, 1.0f));
	//m_pViewport = new nau::render::Viewport();
	m_pCam->setViewport(m_Viewport);
	m_Viewport->setPropf2(Viewport::SIZE, vec2((float)NAU->getWindowWidth(),(float)NAU->getWindowWidth()));
	m_Viewport->setPropf2(Viewport::ORIGIN, vec2(0,0));
	m_pCam->setPrope(Camera::PROJECTION_TYPE, Camera::ORTHO);
	m_pCam->setOrtho(0.0f ,(float)NAU->getWindowWidth() , (float)NAU->getWindowWidth(), 0.0f, -1.0f, 1.0f);

	m_pFont = FontManager::getFont("CourierNew10");

	m_MaterialMap[m_pFont.getMaterialName()] = MaterialID(DEFAULTMATERIALLIBNAME,m_pFont.getMaterialName());

	m_BoolProps[Pass::COLOR_CLEAR] = false;
}


PassProfiler::~PassProfiler(void)
{
}


std::shared_ptr<Pass>
PassProfiler::Create(const std::string &passName) {

	return dynamic_pointer_cast<Pass>(std::shared_ptr<PassProfiler>(new PassProfiler(passName)));
}


void
PassProfiler::setCamera (const std::string &cameraName) {

}


void
PassProfiler::prepare (void)
{
	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->bind();
	}

	RENDERER->pushMatrix(IRenderer::PROJECTION_MATRIX);
	RENDERER->loadIdentity(IRenderer::PROJECTION_MATRIX);

	RENDERER->pushMatrix(IRenderer::VIEW_MATRIX);
	RENDERER->loadIdentity(IRenderer::VIEW_MATRIX);



//	vec3 v = m_pViewport->getSize();

	prepareBuffers();

	m_Viewport->setPropf2(Viewport::SIZE, vec2((float)NAU->getWindowWidth(),(float)NAU->getWindowHeight()));
	m_pCam->setOrtho(0.0f , (float)NAU->getWindowWidth() , (float)NAU->getWindowHeight(), 0.0f, -1.0f, 1.0f);

	if (m_Viewport != NULL) {
		RENDERER->setViewport(m_Viewport);
	}
	RENDERER->setCamera(m_pCam);
	
	RENDERER->loadIdentity(IRenderer::MODEL_MATRIX);
	RENDERER->pushMatrix(IRenderer::MODEL_MATRIX);
	RENDERER->translate(IRenderer::MODEL_MATRIX, vec3(15, 15, 0));

}


void
PassProfiler::restore (void)
{
	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->unbind();
	}

	RENDERER->popMatrix(IRenderer::PROJECTION_MATRIX);
	RENDERER->popMatrix(IRenderer::VIEW_MATRIX);
	RENDERER->popMatrix(IRenderer::MODEL_MATRIX);
}


void 
PassProfiler::doPass (void)
{
//	nau::scene::SceneObject *string;

	std::string prof;
	Profile::DumpLevels(prof);
	m_pFont.createSentenceRenderable(m_pSO->getRenderable(), prof);
	m_pSO->getRenderable()->resetCompilationFlags();
	//m_pSO->setRenderable(r);//Profile::DumpLevels()));
	//string = m_pFont.createSentenceSceneObject(Profile::DumpLevels());
	RENDERMANAGER->clearQueue();
	RENDERMANAGER->addToQueue (m_pSO, m_MaterialMap);
	RENDERMANAGER->processQueue();
}

