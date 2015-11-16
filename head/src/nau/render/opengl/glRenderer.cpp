#include "nau/render/opengl/glRenderer.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
#include "nau/loader/iTextureLoader.h"
#include "nau/material/material.h" 
#include "nau/material/materialGroup.h"
#include "nau/math/matrix.h"
#include "nau/math/vec4.h"
#include "nau/geometry/vertexData.h"
#include "nau/render/opengl/glBuffer.h"
#include "nau/render/opengl/glDebug.h"
#include "nau/render/opengl/glImageTexture.h"
#include "nau/render/opengl/glMaterialGroup.h"
#include "nau/render/opengl/glProgram.h"
#include "nau/render/opengl/glTexture.h"
#include "nau/render/opengl/glVertexArray.h"
#include "nau/render/opengl/glRenderTarget.h"
#include "nau/system/file.h"



using namespace nau::math;
using namespace nau::render;
using namespace nau::render::opengl;
using namespace nau::geometry;
using namespace nau::scene;
using namespace nau::material;

bool GLRenderer::Inited = GLRenderer::Init();

bool
GLRenderer::Init() {

	return true;
}


GLenum GLRenderer::GLPrimitiveTypes[PRIMITIVE_TYPE_COUNT] = 
	{GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_LINES, GL_LINE_LOOP, GL_POINTS, GL_TRIANGLES_ADJACENCY
	, GL_PATCHES
};


GLRenderer::GLRenderer(void) :
	m_TriCounter (0),
	m_glCurrState (),
	m_Lights(),
	m_Camera(NULL),
	m_Textures(),
	m_TexturingFlag(true),
	m_ActualRenderMode(IRenderer::MATERIAL_MODE),
	m_PrevRenderMode(IRenderer::MATERIAL_MODE),
	m_Shader (0)
{
	init();
	GLDebug::Init();
	m_glCurrState.set();
	m_Textures.clear();

	glEnable(GL_MULTISAMPLE);
	registerAndInitArrays(Attribs);
}


GLRenderer::~GLRenderer(void) {

	m_Lights.clear();
}


bool 
GLRenderer::init() {

	glbinding::Binding::initialize(false);

//	glbinding::setCallbackMask(glbinding::CallbackMask::BeforeAndAfter | glbinding::CallbackMask::ParametersAndReturnValue);

	//glewExperimental = true;
	//GLenum error = glewInit();
	//if (GLEW_OK != error){
	//	std::cout << "GLEW init error: " << glewGetErrorString(error) << std::endl;
	//	return false;
	//}
	//else {
		glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &MaxTextureUnits);
		glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &MaxColorAttachments);
		return true;
	//}
}


int 
GLRenderer::setTrace(int frames) {

	return GLDebug::SetTrace(frames);
	// just to be on the safe side
	//if (frames != 0) {
	//	std::string name = nau::system::File::GetCurrentFolder() + "/__nau3Dtrace";
	//	nau::system::File::CreateDir(name);
	//	name += "/Frame_" + std::to_string(RENDERER->getPropui(FRAME_COUNT)) + ".txt";
	//	CLogger::AddLog(CLogger::LEVEL_TRACE, name);
	//	//CLogger::Reset(CLogger::LEVEL_TRACE);
	//}
	


}


// -----------------------------------------------------------------
//		PROPERTIES
// -----------------------------------------------------------------

const mat4 &
GLRenderer::getPropm4(Mat4Property prop) {

	switch (prop) {
	case VIEW_MODEL:
		m_Mat4Props[IRenderer::VIEW_MODEL] = m_Mat4Props[IRenderer::VIEW_MATRIX];
		m_Mat4Props[IRenderer::VIEW_MODEL] *= m_Mat4Props[IRenderer::MODEL_MATRIX];
		return m_Mat4Props[IRenderer::VIEW_MODEL];
	case PROJECTION_VIEW_MODEL:
		m_Mat4Props[IRenderer::PROJECTION_VIEW_MODEL] = m_Mat4Props[IRenderer::PROJECTION_MATRIX];
		m_Mat4Props[IRenderer::PROJECTION_VIEW_MODEL] *= m_Mat4Props[IRenderer::VIEW_MATRIX];
		m_Mat4Props[IRenderer::PROJECTION_VIEW_MODEL] *= m_Mat4Props[IRenderer::MODEL_MATRIX];
		return m_Mat4Props[IRenderer::PROJECTION_VIEW_MODEL];
	case PROJECTION_VIEW:
		m_Mat4Props[IRenderer::PROJECTION_VIEW] = m_Mat4Props[IRenderer::PROJECTION_MATRIX];
		m_Mat4Props[IRenderer::PROJECTION_VIEW] *= m_Mat4Props[IRenderer::VIEW_MATRIX];
		return m_Mat4Props[IRenderer::PROJECTION_VIEW];
	case TS05_PVM:
		m_Mat4Props[IRenderer::TS05_PVM].setIdentity();
		m_Mat4Props[IRenderer::TS05_PVM].translate(0.5f, 0.5f, 0.5f);
		m_Mat4Props[IRenderer::TS05_PVM].scale(0.5f);
		m_Mat4Props[IRenderer::TS05_PVM] *= m_Mat4Props[IRenderer::PROJECTION_MATRIX];
		m_Mat4Props[IRenderer::TS05_PVM] *= m_Mat4Props[IRenderer::VIEW_MATRIX];
		return m_Mat4Props[IRenderer::TS05_PVM];
	default:
		return AttributeValues::getPropm4(prop);
	};
}


const mat3 &
GLRenderer::getPropm3(Mat3Property prop) {

	switch (prop) {
	case NORMAL:
		m_pReturnMatrix = m_Mat4Props[IRenderer::VIEW_MATRIX];
		m_pReturnMatrix *= m_Mat4Props[IRenderer::MODEL_MATRIX];
		m_Mat3Props[IRenderer::NORMAL].setMatrix(m_pReturnMatrix.getSubMat3());
		m_Mat3Props[IRenderer::NORMAL].invert();
		m_Mat3Props[IRenderer::NORMAL].transpose();
		return m_Mat3Props[IRenderer::NORMAL];
	default:
		return AttributeValues::getPropm3(prop);
	}
}


//void *
//GLRenderer::getProp(unsigned int prop, Enums::DataType dt) {
//
//	switch (dt) {
//
//	case Enums::MAT4:
//		switch (prop) {
//		case VIEW_MODEL:
//			m_pReturnMatrix = m_Mat4Props[IRenderer::VIEW_MATRIX];
//			m_pReturnMatrix *= m_Mat4Props[IRenderer::MODEL_MATRIX];
//			return (void *)m_pReturnMatrix.getMatrix();
//		case PROJECTION_VIEW_MODEL:
//			m_pReturnMatrix = m_Mat4Props[IRenderer::PROJECTION_MATRIX];
//			m_pReturnMatrix *= m_Mat4Props[IRenderer::VIEW_MATRIX];
//			m_pReturnMatrix *= m_Mat4Props[IRenderer::MODEL_MATRIX];
//			return (void *)m_pReturnMatrix.getMatrix();
//		case PROJECTION_VIEW:
//			m_pReturnMatrix = m_Mat4Props[IRenderer::PROJECTION_MATRIX];
//			m_pReturnMatrix *= m_Mat4Props[IRenderer::VIEW_MATRIX];
//			return (void *)m_pReturnMatrix.getMatrix();
//		case TS05_PVM:
//			m_pReturnMatrix.setIdentity();
//			m_pReturnMatrix.translate(0.5f, 0.5f, 0.5f);
//			m_pReturnMatrix.scale(0.5f);
//			m_pReturnMatrix *= m_Mat4Props[IRenderer::PROJECTION_MATRIX];
//			m_pReturnMatrix *= m_Mat4Props[IRenderer::VIEW_MATRIX];
//			return (void *)m_pReturnMatrix.getMatrix();
//		default:
//			return AttributeValues::getProp(prop, dt);
//		};
//		break;
//	case Enums::MAT3:
//		switch (prop) {
//		case NORMAL:
//			m_pReturnMatrix = m_Mat4Props[IRenderer::VIEW_MATRIX];
//			m_pReturnMatrix *= m_Mat4Props[IRenderer::MODEL_MATRIX];
//			m_pReturnMat3.setMatrix(m_pReturnMatrix.getSubMat3());
//			m_pReturnMat3.invert();
//			m_pReturnMat3.transpose();
//			return (void *)m_pReturnMat3.getMatrix();
//		default:
//			return AttributeValues::getProp(prop, dt);
//		}
//		break;
//	default:
//		return AttributeValues::getProp(prop, dt);
//	}
//}



// -----------------------------------------------------------------
//		ATOMICS
// -----------------------------------------------------------------


std::vector<unsigned int> &
GLRenderer::getAtomicCounterValues() {

	std::string buffer;
	unsigned int offset;
	unsigned int value;
	int i = 0;
	IBuffer *b;

	if (!APISupport->apiSupport(IAPISupport::BUFFER_ATOMICS)) {
		m_AtomicCounterValues.clear();
		return m_AtomicCounterValues;
	}

	m_AtomicCounterValues.resize(m_AtomicLabels.size());

	if (m_AtomicCount) {

		glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);
		for (auto at : m_AtomicLabels) {
			buffer = at.first.first;
			offset = at.first.second;
			b = RESOURCEMANAGER->getBuffer(buffer);
			if (NULL != b) {
				//GL_ATOMIC_COUNTER_BUFFER
				//b->getData(offset, sizeof(unsigned int), &value);
				glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, b->getPropi(IBuffer::ID));
				glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, offset, sizeof(unsigned int), &value);
				m_AtomicCounterValues[i++] = value;
			}
			else
				m_AtomicCounterValues[i++] = 0;
		}
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	}
	return m_AtomicCounterValues;

}




// -----------------------------------------------------------------
//		LIGHTS
// -----------------------------------------------------------------

unsigned int
GLRenderer::getLightCount() {

	return (unsigned int)m_Lights.size();
}


std::shared_ptr<Light> &
GLRenderer::getLight(unsigned int i) {

	assert(i < m_Lights.size());
	return m_Lights[i];
}


bool
GLRenderer::addLight(std::shared_ptr<Light> &aLight) {

	int id = (unsigned int)m_Lights.size();

	aLight->setPropi(Light::ID, id);
	m_Lights.push_back(aLight);
	m_IntProps[LIGHT_COUNT]++;

	return true;
}

void
GLRenderer::removeLights() {

	for (unsigned int i = 0; i < m_Lights.size(); i++) {
		m_Lights[i]->setPropi(Light::ID, -1);
	}

	m_Lights.clear();
	m_IntProps[LIGHT_COUNT] = 0;
}


// -----------------------------------------------------------------
//		CAMERA
// -----------------------------------------------------------------

void
GLRenderer::setViewport(nau::render::Viewport *aViewport) {

	m_Viewport = aViewport;
	const vec2& vpOrigin = aViewport->getPropf2(Viewport::ABSOLUT_ORIGIN);
	const vec2& vpSize = aViewport->getPropf2(Viewport::ABSOLUT_SIZE);
	const vec4& vpColor = aViewport->getPropf4(Viewport::CLEAR_COLOR);

	glViewport((int)vpOrigin.x, (int)vpOrigin.y, (int)vpSize.x, (int)vpSize.y);
	glClearColor(vpColor.x, vpColor.y, vpColor.z, vpColor.w);

	glEnable(GL_SCISSOR_TEST); // MARK - perform scissor test only if not using the whole window 
	glScissor((int)vpOrigin.x, (int)vpOrigin.y, (int)vpSize.x, (int)vpSize.y);
}


Viewport *
GLRenderer::getViewport() {

	return m_Viewport;
}


void
GLRenderer::setCamera(nau::scene::Camera *aCamera) {

	m_Camera = aCamera;
	setViewport(aCamera->getViewport());

	m_Mat4Props[IRenderer::PROJECTION_MATRIX] = aCamera->getPropm4(Camera::PROJECTION_MATRIX);
	m_Mat4Props[IRenderer::VIEW_MATRIX] = aCamera->getPropm4(Camera::VIEW_MATRIX);
	m_Mat4Props[IRenderer::MODEL_MATRIX].setIdentity();
}


Camera *
GLRenderer::getCamera() {

	return m_Camera;
}


// -----------------------------------------------------------------
//		COUNTERS
// -----------------------------------------------------------------


void
GLRenderer::accumTriCounter(unsigned int drawPrimitive, unsigned int size) {

	switch (drawPrimitive) {
	case (unsigned int)GL_TRIANGLES:
		m_TriCounter += size / 3;
		break;
	case (unsigned int)GL_TRIANGLE_STRIP:
	case (unsigned int)GL_TRIANGLE_FAN:
		m_TriCounter += size - 1;
		break;
	case (unsigned int)GL_LINES:
		m_TriCounter += size / 2;
		break;
	}
}


void
GLRenderer::resetCounters(void) {

	m_TriCounter = 0;
}


unsigned int
GLRenderer::getCounter(Counters c) {

	if (c == TRIANGLE_COUNTER)
		return m_TriCounter;

	return 0;
}


unsigned int
GLRenderer::getNumberOfPrimitives(MaterialGroup *m) {

	unsigned int indices = (unsigned int)m->getIndexData().getIndexSize();
	unsigned int primitive = (unsigned int)m->getParent().getRealDrawingPrimitive();

	switch (primitive) {

	case GL_TRIANGLES_ADJACENCY:
		return (indices / 6);
	case GL_TRIANGLES:
		return (indices / 3);
	case GL_TRIANGLE_STRIP:
	case GL_TRIANGLE_FAN:
		return (indices - 2);
	case GL_LINES:
		return (indices / 2);
	case GL_LINE_LOOP:
		return (indices - 1);
	case GL_POINTS:
		return indices;
	case GL_PATCHES:
		assert(APISupport->apiSupport(IAPISupport::TESSELATION_SHADERS) && "invalid primitive type");
		return indices / m->getParent().getnumberOfVerticesPerPatch();
	default:
		assert(false && "invalid primitive type");
		return (0);
	}
}



// -----------------------------------------------------------------
//		MATRICES
// -----------------------------------------------------------------


void
GLRenderer::loadIdentity(MatrixMode mode) {

	m_Mat4Props[mode].setIdentity();
}


void
GLRenderer::pushMatrix(MatrixMode mode) {

	m_MatrixStack[mode].push_back(m_Mat4Props[mode]);
}


void
GLRenderer::popMatrix(MatrixMode mode) {

	m_Mat4Props[mode] = m_MatrixStack[mode].back();
	m_MatrixStack[mode].pop_back();
}


void
GLRenderer::applyTransform(MatrixMode mode, const nau::math::mat4 &aTransform) {

	m_Mat4Props[mode] *= aTransform;
}


void
GLRenderer::translate(MatrixMode mode, nau::math::vec3 &aVec) {

	m_Mat4Props[mode].translate(aVec);
}


void
GLRenderer::scale(MatrixMode mode, nau::math::vec3 &aVec) {

	m_Mat4Props[mode].scale(aVec);
}


void
GLRenderer::rotate(MatrixMode mode, float angle, nau::math::vec3 &axis) {

	m_Mat4Props[mode].rotate(angle, axis);
}



// -----------------------------------------------------------------
//		MATERIAL - COLOR
// -----------------------------------------------------------------


void
GLRenderer::setMaterial(vec4 &diffuse, vec4 &ambient, vec4 &emission, vec4 &specular, float shininess) {

	m_Material.setPropf(ColorMaterial::SHININESS, shininess);
	m_Material.setPropf4(ColorMaterial::DIFFUSE, diffuse);
	m_Material.setPropf4(ColorMaterial::AMBIENT, ambient);
	m_Material.setPropf4(ColorMaterial::EMISSION, emission);
	m_Material.setPropf4(ColorMaterial::SPECULAR, specular);
}


void
GLRenderer::setMaterial(ColorMaterial &mat) {

	m_Material = mat;
	//m_Material.clone(mat);
}


ColorMaterial *
GLRenderer::getMaterial() {

	return &m_Material;
}


// -----------------------------------------------------------------
//		MATERIAL - STATE
// -----------------------------------------------------------------


void
GLRenderer::setState(IState *aState) {

	m_glCurrState.setDiff(&m_glDefaultState, aState);
}


void
GLRenderer::setDefaultState() {

	m_glCurrState.setDefault();
	m_glDefaultState.setDefault();
	m_glCurrState.set();
}


IState *
GLRenderer::getState() {

	return &m_glCurrState;
}


// -----------------------------------------------------------------
//		MATERIAL - SHADER
// -----------------------------------------------------------------


void
GLRenderer::setShader(IProgram *aShader)
{
	m_Shader = aShader;
}


int
GLRenderer::getAttribLocation(std::string &name) {

	return VertexData::GetAttribIndex(name);
}


// -----------------------------------------------------------------
//		MATERIAL - IMAGE TEXTURE
// -----------------------------------------------------------------



void
GLRenderer::addImageTexture(unsigned int aTexUnit, IImageTexture *t) {

	assert(APISupport->apiSupport(IAPISupport::IMAGE_TEXTURE));
	m_ImageTextures[aTexUnit] = t;
}


void
GLRenderer::removeImageTexture(unsigned int aTexUnit) {

	if (m_ImageTextures.count(aTexUnit))
		m_ImageTextures.erase(aTexUnit);
}


unsigned int
GLRenderer::getImageTextureCount() {

	return (unsigned int)m_ImageTextures.size();
}


IImageTexture*
GLRenderer::getImageTexture(unsigned int aTexUnit) {

	if (m_ImageTextures.count(aTexUnit))
		return m_ImageTextures[aTexUnit];
	else
		return NULL;
}



// -----------------------------------------------------------------
//		MATERIAL - TEXTURE
// -----------------------------------------------------------------


void
GLRenderer::setActiveTextureUnit(unsigned int aTexUnit) {

	glActiveTexture(GL_TEXTURE0 + (int)aTexUnit);
}


void
GLRenderer::addTexture(MaterialTexture *t) {

	unsigned int unit = t->getPropi(MaterialTexture::UNIT);
	m_Textures[unit] = t;
	m_IntProps[TEXTURE_COUNT] |= (1 << unit);
//	m_IntProps[TEXTURE_COUNT]++;
}


void
GLRenderer::removeTexture(unsigned int aTexUnit) {

	if (m_Textures.count(aTexUnit)) {
		m_Textures.erase(aTexUnit);
		m_IntProps[TEXTURE_COUNT] &= ~(1 << aTexUnit) ;
//		m_IntProps[TEXTURE_COUNT]--;
	}
}


MaterialTexture *
GLRenderer::getMaterialTexture(int unit) {

	if (m_Textures.count(unit))
		return m_Textures[unit];
	else
		return NULL;
}


ITexture *
GLRenderer::getTexture(int unit) {

	if (m_Textures.count(unit))
		return m_Textures[unit]->getTexture();
	else
		return NULL;
}


void 
GLRenderer::resetTextures(const std::map<int, MaterialTexture *> &textures)
{
	if (APISupport->getVersion() > 440) {
		glBindTextures(0, MaxTextureUnits, NULL);
		glBindSamplers(0, MaxTextureUnits, NULL);
	}
	else {
		for (auto t : textures) {
			t.second->unbind();
		}
	}
	m_Textures.clear();
	m_IntProps[TEXTURE_COUNT] = 0;
}



// -----------------------------------------------------------------
//		FRAMEBUFFER OPS
// -----------------------------------------------------------------


void
GLRenderer::clearFrameBuffer(unsigned int b) {

	gl::ClearBufferMask c = (gl::ClearBufferMask)0;

	if (b & COLOR_BUFFER) {
		c |= GL_COLOR_BUFFER_BIT;
	}
	if (b & DEPTH_BUFFER) {
		c |= GL_DEPTH_BUFFER_BIT;
	}
	if (b & STENCIL_BUFFER) {
		c |= GL_STENCIL_BUFFER_BIT;
	}

	glClear(c);

	// scissor is enabled when setting the viewport
	glDisable(GL_SCISSOR_TEST);
}


void
GLRenderer::prepareBuffers(Pass *p) {

	int clear = 0;

	bool value = p->getPropb(Pass::DEPTH_ENABLE);
	if (value) {
		glEnable(GL_DEPTH_TEST);
		bool dm = p->getPropb(Pass::DEPTH_MASK);
		glDepthMask((GLboolean)dm);
		m_glDefaultState.setPropb(IState::DEPTH_MASK, dm);
		m_glCurrState.setPropb(IState::DEPTH_MASK, dm);
		GLenum df = translateStencilDepthFunc(p->getPrope(Pass::DEPTH_FUNC));
		glDepthFunc(df);
		m_glDefaultState.setPrope(IState::DEPTH_FUNC, (unsigned int)df);
		m_glCurrState.setPrope(IState::DEPTH_FUNC, (unsigned int)df);
	}
	else
		glDisable(GL_DEPTH_TEST);
	m_glDefaultState.setPropb(IState::DEPTH_TEST, value);
	m_glCurrState.setPropb(IState::DEPTH_TEST, value);

	value = p->getPropb(Pass::STENCIL_ENABLE);

	if (value) {
		glEnable(GL_STENCIL_TEST);
	}
	else
		glDisable(GL_STENCIL_TEST);
	glStencilFunc(translateStencilDepthFunc(p->getPrope(Pass::STENCIL_FUNC)),
		p->getPropi(Pass::STENCIL_OP_REF),
		p->getPropui(Pass::STENCIL_OP_MASK));
	glStencilOp(translateStencilOp(p->getPrope(Pass::STENCIL_FAIL)),
		translateStencilOp(p->getPrope(Pass::STENCIL_DEPTH_FAIL)),
		translateStencilOp(p->getPrope(Pass::STENCIL_DEPTH_PASS)));

	if (p->getPropb(Pass::COLOR_ENABLE)) {
		bvec4 b = bvec4(true, true, true, true);
		m_glDefaultState.setPropb4(IState::COLOR_MASK_B4, b);
	}
	else {
		bvec4 b = bvec4(false, false, false, false);
		m_glDefaultState.setPropb4(IState::COLOR_MASK_B4, b);
	}

	if (p->getPropb(Pass::DEPTH_CLEAR)) {
		clear = IRenderer::DEPTH_BUFFER;
		glClearDepth(p->getPropf(Pass::DEPTH_CLEAR_VALUE));
	}
	if (p->getPropb(Pass::COLOR_CLEAR)) {
		clear |= IRenderer::COLOR_BUFFER;
	}
	if (p->getPropb(Pass::STENCIL_CLEAR)) {
		glClearStencil(p->getPropi(Pass::STENCIL_CLEAR_VALUE));
		clear |= IRenderer::STENCIL_BUFFER;
	}

	clearFrameBuffer(clear);
}


void
GLRenderer::flush(void) {

	glFinish();
}


void
GLRenderer::setDepthClamping(bool b) {

	if (b)
		glEnable(GL_DEPTH_CLAMP);
	else
		glDisable(GL_DEPTH_CLAMP);
}


void
GLRenderer::colorMask(bool r, bool g, bool b, bool a) {

	glColorMask((GLboolean)r, (GLboolean)g, (GLboolean)b, (GLboolean)a);
	bvec4 *bv = new bvec4(r, g, b, a);
	m_glCurrState.setPropb4(IState::COLOR_MASK_B4, *bv);
	m_glDefaultState.setPropb4(IState::COLOR_MASK_B4, *bv);
}


void 
GLRenderer::saveScreenShot() {

	int w, h;
	char *pixels;
	w = NAU->getWindowWidth();
	h = NAU->getWindowHeight();
	pixels = (char *)malloc(w * h * 4);

	glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	
	nau::loader::ITextureLoader::Save(w, h, pixels);
	free (pixels);
}


// -----------------------------------------------------------------
//		RENDER
// -----------------------------------------------------------------


void
GLRenderer::setRenderMode(TRenderMode mode) {

	m_ActualRenderMode = mode;

	switch (mode) {
	case POINT_MODE:
		m_TexturingFlag = false;
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
		break;
	case WIREFRAME_MODE:
		m_TexturingFlag = false;
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		break;
	case SOLID_MODE:
		m_TexturingFlag = false;
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		break;
	case MATERIAL_MODE:
		m_TexturingFlag = true;
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		break;
	}

}


void
GLRenderer::drawGroup(MaterialGroup* aMatGroup) {

	if (aMatGroup->getMaterialName() == "__Emission Green" || aMatGroup->getMaterialName() == "__Emission Red")
		int x = 3;
	IRenderable& aRenderable = aMatGroup->getParent();
	IndexData &indexData = aMatGroup->getIndexData();

	unsigned int drawPrimitive = aRenderable.getRealDrawingPrimitive();

	if (drawPrimitive == GL_PATCHES) {
		assert(APISupport->apiSupport(IAPISupport::TESSELATION_SHADERS));
		int k = aRenderable.getnumberOfVerticesPerPatch();
		glPatchParameteri(GL_PATCH_VERTICES, k);
	}
	// this forces compilation for everything that is rendered!
	// required for animated objects
	if (!aMatGroup->isCompiled())
		aMatGroup->compile();

	GLsizei size;

	{
		PROFILE("Bindings");

		aMatGroup->bind();
	}

	{
		PROFILE_GL("Draw elements");

		size = (GLsizei)indexData.getIndexSize();

		if (size != 0) {
			if (m_UIntProps[IRenderer::BUFFER_DRAW_INDIRECT]) {

				IBuffer *b = RESOURCEMANAGER->getBufferByID(m_UIntProps[IRenderer::BUFFER_DRAW_INDIRECT]); 
				b->setSubData(0, 4, &size);
				unsigned int aux[3] = { 0, 0, 0 };
				//b->setSubData(8, 12, &aux);
				//aux[0] = 1000;
				//b->setSubData(4, 4, &aux);
				b->bind((unsigned int)GL_DRAW_INDIRECT_BUFFER);
				unsigned int temp = 0;
				//glMultiDrawElementsIndirect(drawPrimitive, GL_UNSIGNED_INT, &temp,1,0);
				unsigned int instCount;
				b->getData(4, 4, &instCount);
				//glDrawElementsInstancedBaseVertexBaseInstance(drawPrimitive, size, GL_UNSIGNED_INT, 0, instCount, 0,0);
				glDrawElementsIndirect((GLenum)drawPrimitive, GL_UNSIGNED_INT, 0);
				//glDrawElementsInstanced(drawPrimitive, size, GL_UNSIGNED_INT, 0, instCount);
			}
			else if (m_UIntProps[IRenderer::INSTANCE_COUNT])
				glDrawElementsInstanced((GLenum)drawPrimitive, size, GL_UNSIGNED_INT, 0, m_UIntProps[IRenderer::INSTANCE_COUNT]);
			else
				glDrawElements((GLenum)drawPrimitive, size, GL_UNSIGNED_INT, 0);
		}
		else {
			size = aRenderable.getVertexData().getNumberOfVertices();
						
			if (m_UIntProps[IRenderer::BUFFER_DRAW_INDIRECT]) {

				IBuffer *b = RESOURCEMANAGER->getBufferByID(m_UIntProps[IRenderer::BUFFER_DRAW_INDIRECT]); 
				b->setSubData(0, 4, &size);
				b->bind((unsigned int)GL_DRAW_INDIRECT_BUFFER);
				unsigned int temp = 0;
				glDrawArraysIndirect((GLenum)drawPrimitive, &temp);
			}
			else if (m_UIntProps[IRenderer::INSTANCE_COUNT])
				glDrawArraysInstanced((GLenum)drawPrimitive, 0, size, m_UIntProps[IRenderer::INSTANCE_COUNT]);
			else
				glDrawArrays((GLenum)drawPrimitive, 0, size);
		}
	}

	if (m_BoolProps[DEBUG_DRAW_CALL]) {
		showDrawDebugInfo(aMatGroup);
	}

#ifdef PROFILE
	accumTriCounter(drawPrimitive, size);
#endif

	aMatGroup->unbind();
}


void 
GLRenderer::dispatchCompute(int dimX, int dimY, int dimZ) {

	glDispatchCompute(dimX, dimY, dimZ);

	if (m_BoolProps[DEBUG_DRAW_CALL]) {
		showDrawDebugInfo((PassCompute *)RENDERMANAGER->getCurrentPass());
	}
}


void
GLRenderer::setCullFace(Face aFace) {

	glCullFace(translateFace(aFace));
}


// -----------------------------------------------------------------
//		DEBUG
// -----------------------------------------------------------------


void 
GLRenderer::showDrawDebugInfo(PassCompute *p) {

	nau::util::Tree *t = m_ShaderDebugTree.appendBranch("Pass", p->getName());
	t->appendItem("Class", "Compute");
	Material *mat = p->getMaterial();
	showDrawDebugInfo(mat,t);
	showDrawDebugInfo(mat->getProgram(), t);
}


void
GLRenderer::showDrawDebugInfo(MaterialGroup *mg) {

	Pass *p = RENDERMANAGER->getCurrentPass();
	std::string passName = p->getName();

	nau::util::Tree *tree = &m_ShaderDebugTree;
	if (!tree->hasElement("Pass", passName)) {
		tree = tree->appendBranch("Pass", passName);
		tree->appendItem("Class", p->getClassName());
	}
	else
		tree = tree->getBranch("Pass", passName);

	GLMaterialGroup *glMG = (GLMaterialGroup *)mg;
	std::string s = mg->getMaterialName();
	std::map<string, MaterialID> m = RENDERMANAGER->getCurrentPass()->getMaterialMap();
	if (m.count(s) == 0)
		s = "*";

	std::string meshName = mg->getParent().getName();
	if (!tree->hasElement("Mesh", meshName)) {
		tree = tree->appendBranch("Mesh", meshName);
	}
	else {
		tree = tree->getBranch("Mesh", meshName);
	}

	std::string matName = m[s].getLibName() + "::" + m[s].getMaterialName();
	if (!tree->hasElement("Material", matName))
		tree = tree->appendBranch("Material",  matName);
	else
		tree = tree->getBranch("Material", matName);
	SLOG("Drawing with Material: %s", matName.c_str());

	Material *mat = MATERIALLIBMANAGER->getMaterial(m[s]);

	showDrawDebugInfo(mat, tree);

	SLOG("VAO: %d", glMG->getVAO());
	int buffID;
	buffID = glMG->getIndexData().getBufferID();
	nau::util::Tree *tVAO, *tInputs;
	if (!tree->hasKey("Uniforms and Attributes"))
		tInputs = tree->appendBranch("Uniforms and Attributes");
	else
		tInputs = tree->getBranch("Uniforms and Attributes");

	std::string buffName;
	tVAO = tInputs->appendBranch("VAO", std::to_string(glMG->getVAO()));
	if (buffID) {
		buffName = RESOURCEMANAGER->getBufferByID(buffID)->getLabel();
		tVAO->appendItem("Index Buffer", buffName + " (" + std::to_string(buffID) + ")");
		SLOG("\tIndex Buffer: %d", buffID);
	}
	for (int i = 0; i < VertexData::MaxAttribs; ++i) {
		buffID = glMG->getParent().getVertexData().getBufferID(i);
		if (buffID) {
			buffName = RESOURCEMANAGER->getBufferByID(buffID)->getLabel();
			tVAO->appendItem("Attribute " + VertexData::Syntax[i], " Buffer " + buffName + " (" + std::to_string(buffID) + ")");

			SLOG("\tAttribute %s Buffer: %d", VertexData::Syntax[i].c_str(), buffID);
		}
	}
	showDrawDebugInfo(mat->getProgram(), tVAO);
}


void
GLRenderer::showDrawDebugInfo(Material *mat, nau::util::Tree *tree) {

	std::string programName = mat->getProgram()->getName();
	if (!tree->hasElement("Program", programName))
		tree->appendItem("Program", programName);

	std::vector<unsigned int> vi;
	nau::util::Tree *textureTree, *textureUnitTree;
	mat->getTextureUnits(&vi);
	if (vi.size() > 0) {
		textureTree = tree->appendBranch("Textures");
		SLOG("Textures");
		for (unsigned int i = 0; i < vi.size(); ++i) {
			ITexture *t = mat->getTexture(vi[i]);
			textureUnitTree = textureTree->appendBranch("Unit", to_string(vi[i]));
			textureUnitTree->appendItem("Label", t->getLabel() + " (" + std::to_string(t->getPropi(ITexture::ID)) + ")");
			SLOG("\tUnit %d ID %d Name %s", vi[i], t->getPropi(ITexture::ID), t->getLabel().c_str());
			int id;
			glActiveTexture((GLenum)((int)GL_TEXTURE0+vi[i]));
			glGetIntegerv((GLenum)GLTexture::TextureBound[(GLenum)t->getPrope(ITexture::DIMENSION)], &id);
			textureUnitTree->appendItem("Actual texture bound to unit " + std::to_string(vi[i]), std::to_string(id));
			SLOG("\tActual texture bound to unit %d: %d", vi[i], id);
		}
	}

	vi.clear();
	nau::util::Tree *itTree, *itIDTree;
	mat->getImageTextureUnits(&vi);
	if (vi.size() > 0 && !tree->hasKey("Image Textures")) {
		itTree = tree->appendBranch("Image Textures");
		SLOG("Image Textures");
		for (unsigned int i = 0; i < vi.size(); ++i) {
			GLImageTexture *it = (GLImageTexture *)mat->getImageTexture(vi[i]);
			itIDTree = itTree->appendBranch("Unit", std::to_string(vi[i]));
			GLint k;
			glGetIntegeri_v(GL_IMAGE_BINDING_NAME, (GLuint)vi[i], &k);
			itIDTree->appendItem("Texture ID", std::to_string(k));
			itIDTree->appendItem("Texture Label", RESOURCEMANAGER->getTextureByID(k)->getLabel());
			glGetIntegeri_v(GL_IMAGE_BINDING_LEVEL, (GLuint)vi[i], &k);
			itIDTree->appendItem("Binding level", std::to_string(k));
			glGetIntegeri_v(GL_IMAGE_BINDING_FORMAT, (GLuint)vi[i], &k);
			itIDTree->appendItem("Binding format", GLImageTexture::TexIntFormat[(GLenum)k].name);
			glGetIntegeri_v(GL_IMAGE_BINDING_LAYERED, (GLuint)vi[i], &k);
			if (k) {
				glGetIntegeri_v(GL_IMAGE_BINDING_LAYER, (GLuint)vi[i], &k);
				itIDTree->appendItem("Binding Layer", std::to_string(k));
			}
			glGetIntegeri_v(GL_IMAGE_BINDING_ACCESS, (GLuint)vi[i], &k);
			itIDTree->appendItem("Access", it->Attribs.getListStringOp(GLImageTexture::ACCESS, k));
		}
	}

	vi.clear();
	nau::util::Tree *bufferTree, *bufferIDTree;
	mat->getBufferBindings(&vi);
	if (vi.size() > 0 && !tree->hasKey("Buffers")) {
		bufferTree = tree->appendBranch("Buffers");
		SLOG("Buffers");
		for (unsigned int i = 0; i < vi.size(); ++i) {
			IMaterialBuffer *b = mat->getBuffer(vi[i]);
			IBuffer *buff = b->getBuffer();
			bufferIDTree = bufferTree->appendBranch("Binding Point", std::to_string(vi[i]));
			bufferIDTree->appendItem("Label", buff->getLabel());
			bufferIDTree->appendItem("ID", std::to_string(buff->getPropi(IBuffer::ID)));
			SLOG("\tBinding point %d ID %d Name %s", vi[i], buff->getPropi(IBuffer::ID), buff->getLabel().c_str());
			GLint k;
			glGetIntegeri_v((GLenum)GLBuffer::BufferBound[(GLenum)b->getPrope(IMaterialBuffer::TYPE)], (GLuint)vi[i], &k);
			bufferIDTree->appendItem("Type", b->getAttribSet()->getListStringOp("TYPE", b->getPrope(IMaterialBuffer::TYPE)));
			bufferIDTree->appendItem("Actual buffer bound to unit " + std::to_string(vi[i]), std::to_string(k));
			SLOG("\tActual buffer for binding point %d: %d", vi[i], k);
		}
	}
}



int getUniformByteSize2(int uniSize, 
				int uniType, 
				int uniArrayStride, 
				int uniMatStride) {

	int auxSize;
	if (uniArrayStride > 0)
		auxSize = uniArrayStride * uniSize;
				
	else if (uniMatStride > 0) {

		switch(uniType) {
			case (int)GL_FLOAT_MAT2:
			case (int)GL_FLOAT_MAT2x3:
			case (int)GL_FLOAT_MAT2x4:
			case (int)GL_DOUBLE_MAT2:
			case (int)GL_DOUBLE_MAT2x3:
			case (int)GL_DOUBLE_MAT2x4:
				auxSize = 2 * uniMatStride;
				break;
			case (int)GL_FLOAT_MAT3:
			case (int)GL_FLOAT_MAT3x2:
			case (int)GL_FLOAT_MAT3x4:
			case (int)GL_DOUBLE_MAT3:
			case (int)GL_DOUBLE_MAT3x2:
			case (int)GL_DOUBLE_MAT3x4:
				auxSize = 3 * uniMatStride;
				break;
			case (int)GL_FLOAT_MAT4:
			case (int)GL_FLOAT_MAT4x2:
			case (int)GL_FLOAT_MAT4x3:
			case (int)GL_DOUBLE_MAT4:
			case (int)GL_DOUBLE_MAT4x2:
			case (int)GL_DOUBLE_MAT4x3:
				auxSize = 4 * uniMatStride;
				break;
		}
	}
	else
		auxSize = Enums::getSize(GLUniform::spSimpleType[(GLenum)uniType]);

	return auxSize;
}


void
GLRenderer::showDrawDebugInfo(IProgram *pp, nau::util::Tree *tree) {

	int uniType;
	int activeUnif, actualLen, index, 
		uniSize, uniMatStride, uniArrayStride, uniOffset;
	char name[256];
	char values[512];
	std::string s;

	GLProgram *p = (GLProgram *)pp;
	unsigned int program = p->getProgramID();
	nau::util::Tree *uniformTree;
	uniformTree = tree->appendBranch("Uniforms in default block");
	SLOG("Uniforms Info for program %s (%d)", p->getName().c_str(), program);

	// Get uniforms info (not in named blocks)
	glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &activeUnif);

	for (unsigned int i = 0; i < (unsigned int)activeUnif; ++i) {
		glGetActiveUniformsiv(program, 1, &i, GL_UNIFORM_BLOCK_INDEX, &index);
		if (index == -1) {
			glGetActiveUniformName(program, i, 256, &actualLen, name);	
			glGetActiveUniformsiv(program, 1, &i, GL_UNIFORM_TYPE, &uniType);
			glGetActiveUniformsiv(program, 1, &i, GL_UNIFORM_SIZE, &uniSize);
			glGetActiveUniformsiv(program, 1, &i, GL_UNIFORM_ARRAY_STRIDE, &uniArrayStride);
		
			if (uniType != (int)GL_UNSIGNED_INT_ATOMIC_COUNTER) {
				switch (Enums::getBasicType(GLUniform::spSimpleType[(GLenum)uniType])) {

				case Enums::BOOL:
				case Enums::INT:
				case Enums::SAMPLER:
					glGetUniformiv(program, i, (GLint *)values);
					break;
				case Enums::FLOAT:
					glGetUniformfv(program, i, (GLfloat *)values);
					break;
				case Enums::UINT:
					glGetUniformuiv(program, i, (GLuint *)values);
					break;
				case Enums::DOUBLE:
					glGetUniformdv(program, i, (GLdouble *)values);
					break;
				}
				s = Enums::pointerToString(GLUniform::spSimpleType[(GLenum)uniType], values);
			}
			else s = "";
			int auxSize;
			if (uniArrayStride > 0)
				auxSize = uniArrayStride * uniSize;
			else
				auxSize = Enums::getSize(GLUniform::spSimpleType[(GLenum)uniType]) * uniSize;
			nau::util::Tree *uniTree;
			uniTree = uniformTree->appendBranch(name);
			uniTree->appendItem("Location", std::to_string(i));
			uniTree->appendItem("Type", GLUniform::spGLSLType[(GLenum)uniType]);
			uniTree->appendItem("Size", std::to_string(auxSize));
			uniTree->appendItem("Values", s);
			SLOG("%s (%s) location: %d size: %d values: %s", name, GLUniform::spGLSLType[(GLenum)uniType].c_str(), i,
				auxSize, s.c_str());
		}
	}
	// Get named blocks info
	int count, dataSize, info;
	glGetProgramiv(program, GL_ACTIVE_UNIFORM_BLOCKS, &count);

	for (int i = 0; i < count; ++i) {
		// Get blocks name
		glGetActiveUniformBlockName(program, i, 256, NULL, name);
		nau::util::Tree *uniformTree, *uniTree;
		uniformTree = tree->appendBranch("Block", name);

		glGetActiveUniformBlockiv(program, i, GL_UNIFORM_BLOCK_DATA_SIZE, &dataSize);
		uniformTree->appendItem("Size", std::to_string(dataSize));
		SLOG("%s\n  Size %d", name, dataSize);

		glGetActiveUniformBlockiv(program, i,  GL_UNIFORM_BLOCK_BINDING, &index);
		uniformTree->appendItem("Binding Point", std::to_string(index));
		SLOG("  Block binding point: %d", index);

		glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, index, &info);
		IBuffer *buff = RESOURCEMANAGER->getBufferByID(info);
		if (buff != NULL) {
			uniformTree->appendItem("Buffer", buff->getLabel() + " (" + std::to_string(info) + ")");
			SLOG("  Buffer bound to binding point: %d {", info);
		}
		else
			uniformTree->appendItem("Buffer", "Not Set");


		glGetActiveUniformBlockiv(program, i, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &activeUnif);

		unsigned int *indices;
		indices = (unsigned int *)malloc(sizeof(unsigned int) * activeUnif);
		glGetActiveUniformBlockiv(program, i, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, (int *)indices);
			
		for (int k = 0; k < activeUnif; ++k) {
		
			glGetActiveUniformName(program, indices[k], 256, &actualLen, name);
			glGetActiveUniformsiv(program, 1, &indices[k], GL_UNIFORM_TYPE, &uniType);
			uniTree = uniformTree->appendBranch(name);
			uniTree->appendItem("Type", GLUniform::spGLSLType[(GLenum)uniType]);
			SLOG("\t%s\n\t    %s", name, GLUniform::spGLSLType[(GLenum)uniType].c_str());

			glGetActiveUniformsiv(program, 1, &indices[k], GL_UNIFORM_OFFSET, &uniOffset);
			uniTree->appendItem("Offset", std::to_string(uniOffset));
			SLOG("\t    offset: %d", uniOffset);

			glGetActiveUniformsiv(program, 1, &indices[k], GL_UNIFORM_SIZE, &uniSize);

			glGetActiveUniformsiv(program, 1, &indices[k], GL_UNIFORM_ARRAY_STRIDE, &uniArrayStride);

			glGetActiveUniformsiv(program, 1, &indices[k], GL_UNIFORM_MATRIX_STRIDE, &uniMatStride);

			int auxSize;
			if (uniArrayStride > 0)
				auxSize = uniArrayStride * uniSize;
				
			else if (uniMatStride > 0) {

				switch(uniType) {
					case (int)GL_FLOAT_MAT2:
					case (int)GL_FLOAT_MAT2x3:
					case (int)GL_FLOAT_MAT2x4:
					case (int)GL_DOUBLE_MAT2:
					case (int)GL_DOUBLE_MAT2x3:
					case (int)GL_DOUBLE_MAT2x4:
						auxSize = 2 * uniMatStride;
						break;
					case (int)GL_FLOAT_MAT3:
					case (int)GL_FLOAT_MAT3x2:
					case (int)GL_FLOAT_MAT3x4:
					case (int)GL_DOUBLE_MAT3:
					case (int)GL_DOUBLE_MAT3x2:
					case (int)GL_DOUBLE_MAT3x4:
						auxSize = 3 * uniMatStride;
						break;
					case (int)GL_FLOAT_MAT4:
					case (int)GL_FLOAT_MAT4x2:
					case (int)GL_FLOAT_MAT4x3:
					case (int)GL_DOUBLE_MAT4:
					case (int)GL_DOUBLE_MAT4x2:
					case (int)GL_DOUBLE_MAT4x3:
						auxSize = 4 * uniMatStride;
						break;
				}
			}
			else
				auxSize = Enums::getSize(GLUniform::spSimpleType[(GLenum)uniType]);

			auxSize = getUniformByteSize2(uniSize, uniType, uniArrayStride, uniMatStride);
			uniTree->appendItem("Size", std::to_string(auxSize));
			SLOG("\t    size: %d", auxSize);
			if (uniArrayStride > 0) {
				uniTree->appendItem("Array Stride", std::to_string(uniArrayStride));
				SLOG("\t    array stride: %d", uniArrayStride);
			}
			if (uniMatStride > 0) {
				uniTree->appendItem("Matrix Stride", std::to_string(uniMatStride));
				SLOG("\t    mat stride: %d", uniMatStride);
			}
			if (buff != NULL) {
				buff->getData(uniOffset, auxSize, values);
				s = Enums::valueToStringAligned(GLUniform::spSimpleType[(GLenum)uniType], values);
				uniTree->appendItem("Values", s);
			}
		}
	}
}


// -----------------------------------------------------------------
//		RENDER
// -----------------------------------------------------------------


void
GLRenderer::saveAttrib(IRenderer::RendererAttributes aAttrib) {

	switch (aAttrib) {
	case IRenderer::RENDER_MODE:
		m_PrevRenderMode = m_ActualRenderMode;
		break;
	}
}


void
GLRenderer::restoreAttrib(void) {

	setRenderMode(m_PrevRenderMode);
}


// -----------------------------------------------------------------
//		MISC
// -----------------------------------------------------------------


float
GLRenderer::getDepthAtPoint(int x, int y) {

	float f;
	glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &f);
	return f;
}


nau::math::vec3
GLRenderer::readpixel(int x, int y) {

	GLubyte pixels[3];

	glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pixels);

	vec3 pixel(pixels[0], pixels[1], pixels[2]);

	return pixel;
}


GLenum
GLRenderer::translateStencilDepthFunc(int anOp) {

	GLenum res;

	switch (anOp){
	case Pass::NEVER:
		res = GL_NEVER;
		break;
	case Pass::ALWAYS:
		res = GL_ALWAYS;
		break;
	case Pass::LESS:
		res = GL_LESS;
		break;
	case Pass::LEQUAL:
		res = GL_LEQUAL;
		break;
	case Pass::EQUAL:
		res = GL_EQUAL;
		break;
	case Pass::GEQUAL:
		res = GL_GEQUAL;
		break;
	case Pass::GREATER:
		res = GL_GREATER;
		break;
	case Pass::NOT_EQUAL:
		res = GL_NOTEQUAL;
		break;
	}
	return res;

}


GLenum
GLRenderer::translateStencilOp(int aFunc) {

	switch (aFunc) {
	case Pass::KEEP:
		return GL_KEEP;
	case Pass::ZERO:
		return GL_ZERO;
	case Pass::REPLACE:
		return GL_REPLACE;
	case Pass::INCR:
		return GL_INCR;
	case Pass::INCR_WRAP:
		return GL_INCR_WRAP;
	case Pass::DECR:
		return GL_DECR;
	case Pass::DECR_WRAP:
		return GL_DECR_WRAP;
	case Pass::INVERT:
		return GL_INVERT;
	default:
		return GL_KEEP;
	}
}


GLenum 
GLRenderer::translateFace (Face aFace) {

	switch (aFace) {
		case FRONT:
			return GL_FRONT;
			break;
		//case FRONT_AND_BACK:
		//	return GL_FRONT_AND_BACK;
		//	break;
		case BACK:
			return GL_BACK;
			break;
	    default:
		  return GL_INVALID_ENUM;
	}
}


unsigned int 
GLRenderer::translateDrawingPrimitive (unsigned int aDrawPrimitive) {

	if (IRenderer::PRIMITIVE_TYPE_COUNT > aDrawPrimitive)
		return (unsigned int)GLPrimitiveTypes[aDrawPrimitive];
	else
		return (unsigned int)GL_INVALID_ENUM;	
}

