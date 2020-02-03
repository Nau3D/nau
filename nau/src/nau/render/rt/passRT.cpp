#include "nau/config.h"

#if NAU_RT == 1

#include "nau/render/rt/passRT.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
#include "nau/render/passFactory.h"
#include "nau/render/rt/rtException.h"
#include "nau/system/file.h"


using namespace nau::material;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::render::rt;
using namespace nau::geometry;

bool PassRT::Inited = PassRT::Init();

bool
PassRT::Init() {

	PASSFACTORY->registerClass("rt", Create);
	return true;
}


PassRT::PassRT(const std::string& passName) :
	Pass(passName) , m_RTisReady(false) {

	m_VertexAttributes.resize(VertexData::MaxAttribs);
	for (int i = 0; i < VertexData::MaxAttribs; ++i)
		m_VertexAttributes[i] = false;

	m_RTisReady = false;
	m_RThasIssues = true;
	bool res = RTRenderer::Init();
	if (res)
		m_RThasIssues = false;
}

PassRT::~PassRT() {

}


std::shared_ptr<Pass>
PassRT::Create(const std::string& passName) {

	return dynamic_pointer_cast<Pass>(std::shared_ptr<PassRT>(new PassRT(passName)));
}


void
PassRT::addRayType(const std::string& name) {

	m_ProgramManager.addRayType(name);
}


void
PassRT::addVertexAttribute(const unsigned int i) {

	m_VertexAttributes[i] = true;
}


void 
PassRT::rtInit() {
	if (!m_ProgramManager.generateModules()) {
		m_RThasIssues = true;
		return ;
	}

	if (!m_ProgramManager.generatePrograms()) {
		m_RThasIssues = true;
		return;
	}
	if (!m_ProgramManager.generatePipeline()) {
		m_RThasIssues = true;
		return;
	}
	if (!m_ProgramManager.generateSBT()) {
		m_RThasIssues = true;
		return;
	}

	m_RTisReady = true;
}


void 
PassRT::prepare(void) {

	if (!m_RTisReady && !m_RThasIssues)
		rtInit();
}


void 
PassRT::setRayGenProcedure(const std::string &file, const std::string &proc) {
	m_ProgramManager.setRayGenProcedure(file, proc);
}


void 
PassRT::setDefaultProc(const std::string& pRayType, int procType, const std::string& pFile, const std::string& pName) {

	m_ProgramManager.setDefaultProc(pRayType, procType, pFile, pName);
}



void 
PassRT::addScene(const std::string& sceneName) {

}


void 
PassRT::setRenderTarget(nau::render::IRenderTarget* rt) {

	// do I havo to use this? compiler complains if no namespace is defined ...
	GLuint k = (GLuint)gl::glGetError();
	if (rt == NULL) {
		//if (m_RenderTarget != NULL) 
		//	delete m_Viewport;
		m_UseRT = true;
	}
	else {
		if (m_RenderTarget == NULL) {
			std::string s = "__" + m_Name;
			m_Viewport = RENDERMANAGER->createViewport(s);
			m_UseRT = true;
		}
		setRTSize(rt->getPropui2(IRenderTarget::SIZE));
		m_Viewport->setPropf4(Viewport::CLEAR_COLOR, rt->getPropf4(IRenderTarget::CLEAR_VALUES));
	}
	m_RenderTarget = rt;

	unsigned int n = rt->getNumberOfColorTargets();

	m_OutputPBO.resize(n);
	gl::glGenBuffers(n, (unsigned int*)&m_OutputPBO[0]);

	m_OutputCGR.resize(n);
	m_OutputTexIDs.resize(n);
	m_OutputBufferPrs.resize(n);

	nau::material::ITexture* tex;

	try {
		for (unsigned int i = 0; i < n; ++i) {

			tex = rt->getTexture(i);
			m_OutputTexIDs[i] = tex->getPropi(ITexture::ID);
			int format = tex->getPrope(ITexture::FORMAT);

			gl::glBindBuffer(gl::GL_PIXEL_UNPACK_BUFFER, m_OutputPBO[i]);
			// need to allow different types
			nau::math::uivec2 vec2;
			vec2 = rt->getPropui2(IRenderTarget::SIZE);
			gl::glBufferData(gl::GL_PIXEL_UNPACK_BUFFER, vec2.x * vec2.y * rt->getTexture(i)->getPropi(ITexture::ELEMENT_SIZE) / 8, 0, gl::GL_STREAM_READ);
			gl::glBindBuffer(gl::GL_PIXEL_UNPACK_BUFFER, 0);

			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&(m_OutputCGR[i]), m_OutputPBO[i], cudaGraphicsRegisterFlagsWriteDiscard));
		}
	}
	catch (std::exception const& e) {
		SLOG("Exception setting render target: %s", e.what());
		m_RThasIssues = true;
	}	
	gl::glFinish();
}


void 
PassRT::restore(void) {}
void 
PassRT::doPass(void) {

	if (m_RThasIssues || !m_RTisReady)
		return;

	try {
		size_t size;
		for (int i = 0; i < m_OutputPBO.size(); ++i) {
			// map resources from OpenGL
			CUDA_CHECK(cudaGraphicsMapResources(1, &m_OutputCGR[i], RTRenderer::getOptixStream()));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&m_OutputBufferPrs[i], &size, m_OutputCGR[i]));
		}

		// update launch params
		launchParams.frame = RENDERER->getPropui(IRenderer::FRAME_COUNT);
		launchParams.fbSize.x = 1024; launchParams.fbSize.y = 1024;
		launchParams.colorBuffer = (uint32_t*)m_OutputBufferPrs[0];

		CUdeviceptr d_param;
		// malloc needs to be done only once
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(launchParams)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &launchParams, sizeof(launchParams), cudaMemcpyHostToDevice));

		// render
		OPTIX_CHECK(optixLaunch(
			m_ProgramManager.getPipeline(),
			RTRenderer::getOptixStream(),
			d_param,
			sizeof(launchParams),
			&m_ProgramManager.getSBT(), 1024, 1024, 1));

		
		CUDA_SYNC_CHECK();


		for (int i = 0; i < m_OutputPBO.size(); ++i) {

			// unmap resources from OpenGL
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_OutputCGR[i], RTRenderer::getOptixStream()))

				// copy buffer to texture
				gl::glBindTexture((gl::GLenum)GL_TEXTURE_2D, m_OutputTexIDs[i]);
				gl::glBindBuffer(gl::GL_PIXEL_UNPACK_BUFFER, m_OutputPBO[i]);
				gl::glPixelStorei((gl::GLenum)GL_UNPACK_ALIGNMENT, 4);
				gl::glTexSubImage2D((gl::GLenum)GL_TEXTURE_2D, 0, 0, 0,
					1024, 1024,
					(gl::GLenum)GL_RGBA,
					(gl::GLenum)GL_UNSIGNED_BYTE, 0);
		}
	}
	catch (std::exception const& e) {
		SLOG("Exception when rendering: %s", e.what());
		m_RThasIssues = true;
	}
}

#endif