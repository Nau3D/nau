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
		m_ProgramManager.unregisterTexture(m_RenderTarget->getTexture(0)->getPropi(ITexture::ID));
	}

	if (0 != m_RenderTarget && true == m_UseRT) {

		if (m_ExplicitViewport) {
			vec2 f2 = m_Viewport[0]->getPropf2(Viewport::ABSOLUTE_SIZE);
			if (m_RTSizeWidth != (int)f2.x || m_RTSizeHeight != (int)f2.y) {
				m_RTSizeWidth = (int)f2.x;
				m_RTSizeHeight = (int)f2.y;
				uivec2 uiv2((unsigned int)m_RTSizeWidth, (unsigned int)m_RTSizeHeight);

				m_RenderTarget->setPropui2(IRenderTarget::SIZE, uiv2);
				//m_ProgramManager.registerTexture(m_RenderTarget->getTexture(0)->getPropi(ITexture::ID));
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
PassRT::setMatProc(const std::string& matName, const std::string& pRayType, int procType, const std::string& pFile, const std::string& pName) {

	m_ProgramManager.setMatProc(matName, pRayType, procType, pFile, pName);
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

		if (p.type == "TEXTURE" && p.component == "ID") {

			p.size = sizeof(cudaTextureObject_t);
			p.id = RESOURCEMANAGER->getTexture(p.context)->getPropi(ITexture::ID);
		}
		else if (p.type == "BUFFER" && p.component == "ID") {
			p.size = sizeof(unsigned char *);
			IBuffer* b = RESOURCEMANAGER->getBuffer(p.context);
			p.id = b->getPropi(IBuffer::ID);
			m_BuffersCGR[p.id] = nullptr; 
			m_BuffersPtrs[p.id] = nullptr;
			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&(m_BuffersCGR[p.id]), p.id, cudaGraphicsRegisterFlagsWriteDiscard));
			CUDA_CHECK(cudaGraphicsMapResources(1, &m_BuffersCGR[p.id], RTRenderer::getOptixStream()));
		}

		else {
			AttribSet* attrSet = NAU->getAttribs(p.type);

			attrSet->getPropTypeAndId(p.component, &p.dt, &attr);
			p.offset = 0;
			p.size = Enums::getSize(p.dt);
			p.attr = attr;
		}

		count += p.size;
	}
	return count;
}


void 
PassRT::copyParamsToBuffer() {

	char* temp = (char*)malloc(m_ParamsSize);
	int currOffset = 0;
	size_t size;
	for (auto& p : m_Params) {

		if (p.type == "TEXTURE" && p.component == "ID") {

			memcpy(temp + currOffset, (void *)&m_ProgramManager.m_Textures[p.id].cto, p.size);
		}
		else if (p.type == "BUFFER" && p.component == "ID") {

			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&m_BuffersPtrs[p.id], &size, m_BuffersCGR[p.id]));
			memcpy(temp + currOffset, (void *)&m_BuffersPtrs[p.id], p.size);
		}
		else {

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
			void* d;
			switch (p.dt) {
			case Enums::INT:
			case Enums::BOOL:
			case Enums::FLOAT:
			case Enums::UINT:
			case Enums::SAMPLER:
			case Enums::DOUBLE:
				d = values;
				break;
			default:
				d = ((Data*)values)->getPtr();
			}
			memcpy(temp + currOffset, d, p.size);
		}
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
			m_Viewport.push_back(RENDERMANAGER->createViewport(s));
			m_UseRT = true;
		}
		setRTSize(rt->getPropui2(IRenderTarget::SIZE));
		m_Viewport[0]->setPropf4(Viewport::CLEAR_COLOR, rt->getPropf4(IRenderTarget::CLEAR_VALUES));
	}
	m_RenderTarget = rt;

	bindCudaRenderTarget();
}


void 
PassRT::restore(void) {

	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->unbind();
	}

	restoreCamera();
	RENDERER->removeLights();
}


bool compare(const float3 &a, const float3 &b) {

	if ((a.x != b.x) || (a.y != b.y) || (a.z != b.z))
		return false;
	return true;
}

void
PassRT::setupCamera(void) {

	float ratio;
	std::shared_ptr<Camera>& aCam = RENDERMANAGER->getCamera(m_StringProps[CAMERA]);
	m_LaunchSize = m_RenderTarget->getPropui2(IRenderTarget::SIZE);

	if (m_ExplicitViewport) {
		m_RestoreViewport = aCam->getViewport();
		aCam->setViewport(m_Viewport[0]);
		ratio = m_Viewport[0]->getPropf(Viewport::ABSOLUTE_RATIO);
	}
	else {
		ratio = (float)m_LaunchSize.x / m_LaunchSize.y;
	}
	float fov = aCam->getPropf(Camera::FOV) * 0.5f;
	float fovTan = tanf(DegToRad(fov));

	launchParams.camera.changed = false;

	vec4 pos = aCam->getPropf4(Camera::POSITION);
	const float3 cpos = make_float3(pos.x, pos.y, pos.z);
	if (!compare(cpos, launchParams.camera.position)) {

		launchParams.camera.position = cpos;
		launchParams.camera.changed = true;
	}

	vec4 dir = aCam->getPropf4(Camera::NORMALIZED_VIEW_VEC);

	const float3 cdir = make_float3(dir.x, dir.y, dir.z);
	if (!compare(cdir, launchParams.camera.direction)) {

		launchParams.camera.direction = cdir;
		launchParams.camera.changed = true;
	}

	vec4 up = aCam->getPropf4(Camera::NORMALIZED_UP_VEC);
	up *= fovTan;
	const float3 cup = make_float3(up.x, up.y, up.z);
	if (!compare(cup, launchParams.camera.vertical)) {

		launchParams.camera.vertical = cup;
		launchParams.camera.changed = true;
	}

	vec4 right = aCam->getPropf4(Camera::NORMALIZED_RIGHT_VEC);
	if (ratio != 0)
		right *= fovTan * ratio;
	const float3 cright = make_float3(right.x, right.y, right.z);
	if (!compare(cright, launchParams.camera.horizontal)) {

		launchParams.camera.horizontal = cright;
		launchParams.camera.changed = true;
	}

	if (launchParams.camera.changed)
		launchParams.frame.subFrame = 0;
	else
		launchParams.frame.subFrame++;


	RENDERER->setCamera(aCam, m_Viewport);
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
		if (launchParams.frame.frame == 0)
			launchParams.frame.subFrame = 0;
		launchParams.frame.colorBuffer = (uint32_t*)m_OutputBufferPrs[0];
		//launchParams.frame.accumBuffer = (float4*)m_BuffersPtrs[3];
		const int rpp = getPropi(Pass::RAYS_PER_PIXEL);
		if (launchParams.frame.raysPerPixel != rpp) {
			launchParams.frame.raysPerPixel = rpp;
			launchParams.frame.subFrame = 0;
		}
		const int md = getPropi(Pass::MAX_DEPTH);
		if (launchParams.frame.maxDepth != md) {
			launchParams.frame.maxDepth = md;
			launchParams.frame.subFrame = 0;
		}

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

				//int info;
				//glGetTexLevelParameteriv((gl::GLenum)GL_TEXTURE_2D, 0,
				//	(gl::GLenum)GL_TEXTURE_HEIGHT, &info);

				gl::glBindBuffer(gl::GL_PIXEL_UNPACK_BUFFER, m_OutputPBO[i]);
				gl::glPixelStorei((gl::GLenum)GL_UNPACK_ALIGNMENT, 1);
				gl::glTexSubImage2D((gl::GLenum)GL_TEXTURE_2D, 0, 0, 0,
					m_LaunchSize.x, m_LaunchSize.y,
					(gl::GLenum)GL_RGBA,
					(gl::GLenum)GL_UNSIGNED_BYTE, 0);
				if (getPropb(COLOR_CLEAR) || launchParams.frame.subFrame == 0)
					gl::glClearBufferData(gl::GL_PIXEL_UNPACK_BUFFER, gl::GL_R8, (gl::GLenum)GL_RED, (gl::GLenum)GL_UNSIGNED_BYTE, NULL);
				gl::glBindBuffer(gl::GL_PIXEL_UNPACK_BUFFER, 0);
		}
	}
	catch (std::exception const& e) {
		SLOG("Exception when rendering: %s", e.what());
		m_RThasIssues = true;
	}
}


#endif  