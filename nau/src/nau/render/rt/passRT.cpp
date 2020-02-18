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
	m_RThasIssues = false;

	m_ClassName = "rt";
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
PassRT::rtInit() {

	if (!RTRenderer::Init())
		m_RThasIssues = true;

	if (!m_Geometry.generateAccel(m_SceneVector)) {
		m_RThasIssues = true;
		return;
	}
	launchParams.traversable = m_Geometry.getTraversableHandle();
	
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

	if (!m_ProgramManager.processTextures()) {
		m_RThasIssues = true;
		return;
	}

	if (!m_ProgramManager.generateSBT(m_Geometry.getCudaVBOS())) {
		m_RThasIssues = true;
		return;
	}

	m_LaunchParamsBuffer.setSize(sizeof(launchParams));
	m_RTisReady = true;
}


void 
PassRT::prepare(void) {

	if (!m_RTisReady && !m_RThasIssues) {
		rtInit();
		m_ParamsSize = computeParamsByteSize();
		m_ParamsBuffer.setSize(m_ParamsSize);
	}

	if (0 != m_RenderTarget && true == m_UseRT) {

		if (m_ExplicitViewport) {
			vec2 f2 = m_Viewport->getPropf2(Viewport::ABSOLUTE_SIZE);
			if (m_RTSizeWidth!= (int)f2.x || m_RTSizeHeight != (int)f2.y) {
				m_RTSizeWidth = (int)f2.x;
				m_RTSizeHeight = (int)f2.y;
				uivec2 uiv2((unsigned int)m_RTSizeWidth, (unsigned int)m_RTSizeHeight);
				m_RenderTarget->setPropui2(IRenderTarget::SIZE, uiv2);
				bindCudaRenderTarget();
			}
		}
	}

	setupCamera();
	setupLights();
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

	if (m_SceneVector.end() == std::find(m_SceneVector.begin(), m_SceneVector.end(), sceneName)) {

		m_SceneVector.push_back(sceneName);

		const std::set<std::string>& materialNames =
			RENDERMANAGER->getScene(sceneName)->getMaterialNames();

		for (auto iter : materialNames) {

			if (m_MaterialMap.count(iter) == 0)
				m_MaterialMap[iter] = MaterialID(DEFAULTMATERIALLIBNAME, iter);
		}

		std::shared_ptr<IScene>& sc = RENDERMANAGER->getScene(sceneName);
		sc->compile();
	}
}


void 
PassRT::addVertexAttribute(unsigned int  attr) {

	m_Geometry.addVertexAttribute(attr);
}



void 
PassRT::addParam(const std::string& name, const std::string& type, const std::string& context, const std::string& component, int id) {

	Param p;
	p.name = name;
	p.type = type;
	p.context = context;
	p.component = component;
	p.id = id;

	p.offset = 0;
	p.size = 0;

	m_Params.push_back(p);
}


int 
PassRT::computeParamsByteSize() {

	int count = 0;
	int attr;



	for (auto& p : m_Params) {

		AttribSet* attrSet = NAU->getAttribs(p.type);

		attrSet->getPropTypeAndId(p.component, &p.dt, &attr);
		p.offset = 0;
		p.size = Enums::getSize(p.dt);
		p.attr = attr;
		count += p.size;
	}
	return count;
}


void 
PassRT::copyParamsToBuffer() {

	char* temp = (char*)malloc(m_ParamsSize);
	int currOffset = 0;
	for (auto& p : m_Params) {

		AttributeValues* attr = NULL;
		if (p.context != "CURRENT") {
			attr = NAU->getObjectAttributes(p.type, p.context, p.id);
		}
		else {
			attr = NAU->getCurrentObjectAttributes(p.type, p.id);
		}

		void* values;
		if (attr != NULL) {
			values = attr->getProp(p.attr, p.dt);
		}
		void* d = (Data *)((Data*)values)->getPtr();
		memcpy(temp , d, p.size);
		
		currOffset += p.size;
	}
	m_ParamsBuffer.copy(temp);
	free(temp);
}


void 
PassRT::cleanCudaRenderTargetBindings() {

	//clean up previous PBOs
	try {
		for (int i = 0; i < m_OutputPBO.size(); ++i) {
			CUDA_CHECK(cudaGraphicsUnregisterResource(m_OutputCGR[i]));
			gl::glDeleteBuffers(1, &m_OutputPBO[i]);
		}
	}
	catch (std::exception const& e) {
		SLOG("Exception cleaning render targets: %s", e.what());
		//m_RThasIssues = true;
	}

}


void
PassRT::bindCudaRenderTarget() {

	cleanCudaRenderTargetBindings();

	unsigned int n = m_RenderTarget->getNumberOfColorTargets();

	m_OutputPBO.resize(n);
	m_OutputCGR.resize(n);
	m_OutputTexIDs.resize(n);
	m_OutputBufferPrs.resize(n);

	nau::material::ITexture* tex;

	try {
		gl::glGenBuffers(n, (unsigned int*)&m_OutputPBO[0]);

		for (unsigned int i = 0; i < n; ++i) {

			tex = m_RenderTarget->getTexture(i);
			m_OutputTexIDs[i] = tex->getPropi(ITexture::ID);
			int format = tex->getPrope(ITexture::FORMAT);

			gl::glBindBuffer(gl::GL_ARRAY_BUFFER, m_OutputPBO[i]);
			// need to allow different types
			nau::math::uivec2 vec2;
			vec2 = m_RenderTarget->getPropui2(IRenderTarget::SIZE);
			gl::glBufferData(gl::GL_ARRAY_BUFFER, vec2.x * vec2.y * m_RenderTarget->getTexture(i)->getPropi(ITexture::ELEMENT_SIZE) / 8, 0, gl::GL_STREAM_READ);
			gl::glBindBuffer(gl::GL_ARRAY_BUFFER, 0);

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

	bindCudaRenderTarget();
}


void 
PassRT::restore(void) {}

void
PassRT::setupCamera(void) {

	float ratio;
	std::shared_ptr<Camera>& aCam = RENDERMANAGER->getCamera(m_StringProps[CAMERA]);
	m_LaunchSize = m_RenderTarget->getPropui2(IRenderTarget::SIZE);

	if (m_ExplicitViewport) {
		m_RestoreViewport = aCam->getViewport();
		aCam->setViewport(m_Viewport);
		ratio = m_Viewport->getPropf(Viewport::ABSOLUTE_RATIO);
	}
	else {
		ratio = (float)m_LaunchSize.x / m_LaunchSize.y;
	}
	float fov = aCam->getPropf(Camera::FOV) * 0.5f;
	float fovTan = tanf(DegToRad(fov));

	vec4 pos = aCam->getPropf4(Camera::POSITION);
	launchParams.camera.position = make_float3(pos.x, pos.y, pos.z);

	vec4 dir = aCam->getPropf4(Camera::NORMALIZED_VIEW_VEC);
	launchParams.camera.direction = make_float3(dir.x, dir.y, dir.z);

	vec4 up = aCam->getPropf4(Camera::NORMALIZED_UP_VEC);
	up *= fovTan;
	launchParams.camera.vertical = make_float3(up.x, up.y, up.z);

	vec4 right = aCam->getPropf4(Camera::NORMALIZED_RIGHT_VEC);
	if (ratio != 0)
		right *= fovTan * ratio;
	launchParams.camera.horizontal = make_float3(right.x, right.y, right.z);

	RENDERER->setCamera(aCam);
}


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

		// copy global params to buffer
		copyParamsToBuffer();

		// update launch params
		launchParams.frame.frame = RENDERER->getPropui(IRenderer::FRAME_COUNT);
		launchParams.frame.colorBuffer = (uint32_t*)m_OutputBufferPrs[0];
		launchParams.frame.raysPerPixel = getPropi(Pass::RAYS_PER_PIXEL);
		launchParams.globalParams = (optixParams *)m_ParamsBuffer.getPtr();
		
		launchParams.traversable = m_Geometry.getTraversableHandle();

		m_LaunchParamsBuffer.copy((void *)&launchParams);

		CUDA_SYNC_CHECK();

		// render
		OPTIX_CHECK(optixLaunch(
			m_ProgramManager.getPipeline(),
			RTRenderer::getOptixStream(),
			m_LaunchParamsBuffer.getPtr(),
			m_LaunchParamsBuffer.getSize(),
			&m_ProgramManager.getSBT(), m_LaunchSize.x, m_LaunchSize.y, 1));
		
		CUDA_SYNC_CHECK();

		for (int i = 0; i < m_OutputPBO.size(); ++i) {

			// unmap resources from OpenGL
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_OutputCGR[i], RTRenderer::getOptixStream()))

				// copy buffer to texture
				gl::glBindTexture((gl::GLenum)GL_TEXTURE_2D, m_OutputTexIDs[i]);
				gl::glBindBuffer(gl::GL_PIXEL_UNPACK_BUFFER, m_OutputPBO[i]);
				gl::glPixelStorei((gl::GLenum)GL_UNPACK_ALIGNMENT, 1);
				gl::glTexSubImage2D((gl::GLenum)GL_TEXTURE_2D, 0, 0, 0,
					m_LaunchSize.x, m_LaunchSize.y,
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