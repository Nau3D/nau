#include "nau/render/opengl/glrenderer.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
#include "nau/material/material.h" 
#include "nau/material/materialgroup.h"
#include "nau/math/matrix.h"
#include "nau/math/vec4.h"
#include "nau/render/opengl/glvertexarray.h"
#include "nau/render/opengl/glrendertarget.h"

using namespace nau::math;
using namespace nau::render;
using namespace nau::geometry;
using namespace nau::scene;
using namespace nau::material;

bool GLRenderer::Inited = GLRenderer::Init();

bool
GLRenderer::Init() {

	return true;
}


unsigned int GLRenderer::GLPrimitiveTypes[PRIMITIVE_TYPE_COUNT] = 
	{GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_LINES, GL_LINE_LOOP, GL_POINTS, GL_TRIANGLES_ADJACENCY
#if NAU_OPENGL_VERSION >= 400	
	, GL_PATCHES
#endif
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
	m_glCurrState.set();
	m_Textures.clear();

	glEnable(GL_MULTISAMPLE);
	registerAndInitArrays("RENDERER", Attribs);
}


GLRenderer::~GLRenderer(void) {

	m_Lights.clear();
}


bool 
GLRenderer::init() {

	glewExperimental = true;
	GLenum error = glewInit();
	if (GLEW_OK != error){
		std::cout << "GLEW init error: " << glewGetErrorString(error) << std::endl;
		return false;
	}
	else {
		glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &MaxTextureUnits);
		glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &MaxColorAttachments);
		return true;
	}
}


// -----------------------------------------------------------------
//		PROPERTIES
// -----------------------------------------------------------------

const mat4 &
GLRenderer::getPropm4(Mat4Property prop) {

	switch (prop) {
	case VIEW_MODEL:
		m_pReturnMatrix = m_Mat4Props[IRenderer::VIEW_MATRIX];
		m_pReturnMatrix *= m_Mat4Props[IRenderer::MODEL_MATRIX];
		return m_pReturnMatrix;
	case PROJECTION_VIEW_MODEL:
		m_pReturnMatrix = m_Mat4Props[IRenderer::PROJECTION_MATRIX];
		m_pReturnMatrix *= m_Mat4Props[IRenderer::VIEW_MATRIX];
		m_pReturnMatrix *= m_Mat4Props[IRenderer::MODEL_MATRIX];
		return m_pReturnMatrix;
	case PROJECTION_VIEW:
		m_pReturnMatrix = m_Mat4Props[IRenderer::PROJECTION_MATRIX];
		m_pReturnMatrix *= m_Mat4Props[IRenderer::VIEW_MATRIX];
		return m_pReturnMatrix;
	case TS05_PVM:
		m_pReturnMatrix.setIdentity();
		m_pReturnMatrix.translate(0.5f, 0.5f, 0.5f);
		m_pReturnMatrix.scale(0.5f);
		m_pReturnMatrix *= m_Mat4Props[IRenderer::PROJECTION_MATRIX];
		m_pReturnMatrix *= m_Mat4Props[IRenderer::VIEW_MATRIX];
		return m_pReturnMatrix;
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
		m_pReturnMat3.setMatrix(m_pReturnMatrix.getSubMat3());
		m_pReturnMat3.invert();
		m_pReturnMat3.transpose();
		return m_pReturnMat3;
	default:
		return AttributeValues::getPropm3(prop);
	}
}


void *
GLRenderer::getProp(unsigned int prop, Enums::DataType dt) {

	switch (dt) {

	case Enums::MAT4:
		switch (prop) {
		case VIEW_MODEL:
			m_pReturnMatrix = m_Mat4Props[IRenderer::VIEW_MATRIX];
			m_pReturnMatrix *= m_Mat4Props[IRenderer::MODEL_MATRIX];
			return (void *)m_pReturnMatrix.getMatrix();
		case PROJECTION_VIEW_MODEL:
			m_pReturnMatrix = m_Mat4Props[IRenderer::PROJECTION_MATRIX];
			m_pReturnMatrix *= m_Mat4Props[IRenderer::VIEW_MATRIX];
			m_pReturnMatrix *= m_Mat4Props[IRenderer::MODEL_MATRIX];
			return (void *)m_pReturnMatrix.getMatrix();
		case PROJECTION_VIEW:
			m_pReturnMatrix = m_Mat4Props[IRenderer::PROJECTION_MATRIX];
			m_pReturnMatrix *= m_Mat4Props[IRenderer::VIEW_MATRIX];
			return (void *)m_pReturnMatrix.getMatrix();
		case TS05_PVM:
			m_pReturnMatrix.setIdentity();
			m_pReturnMatrix.translate(0.5f, 0.5f, 0.5f);
			m_pReturnMatrix.scale(0.5f);
			m_pReturnMatrix *= m_Mat4Props[IRenderer::PROJECTION_MATRIX];
			m_pReturnMatrix *= m_Mat4Props[IRenderer::VIEW_MATRIX];
			return (void *)m_pReturnMatrix.getMatrix();
		default:
			return AttributeValues::getProp(prop, dt);
		};
		break;
	case Enums::MAT3:
		switch (prop) {
		case NORMAL:
			m_pReturnMatrix = m_Mat4Props[IRenderer::VIEW_MATRIX];
			m_pReturnMatrix *= m_Mat4Props[IRenderer::MODEL_MATRIX];
			m_pReturnMat3.setMatrix(m_pReturnMatrix.getSubMat3());
			m_pReturnMat3.invert();
			m_pReturnMat3.transpose();
			return (void *)m_pReturnMat3.getMatrix();
		default:
			return AttributeValues::getProp(prop, dt);
		}
		break;
	default:
		return AttributeValues::getProp(prop, dt);
	}
}



// -----------------------------------------------------------------
//		ATOMICS
// -----------------------------------------------------------------

#if (NAU_OPENGL_VERSION >= 400)


std::vector<unsigned int> &
GLRenderer::getAtomicCounterValues() {

	std::string buffer;
	unsigned int offset;
	unsigned int value;
	int i = 0;
	IBuffer *b;

	m_AtomicCounterValues.resize(m_AtomicLabels.size());

	if (m_AtomicCount) {

		//glFinish();
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

//void 
//GLRenderer::prepareAtomicCounterBuffer() {
//
//	m_AtomicCounterValues = (unsigned int *)malloc(sizeof(unsigned int)*(m_AtomicMaxID+1));
//	//IBuffer *b = RESOURCEMANAGER->createBuffer()
//	glGenBuffers(1, &m_AtomicCountersBuffer);
//	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCountersBuffer);
//	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint) * (m_AtomicMaxID + 1), NULL, GL_DYNAMIC_DRAW);
//	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
//	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, m_AtomicCountersBuffer);
//	m_AtomicBufferPrepared = true;
//
//}

//void 
//GLRenderer::resetAtomicCounters() {
//
//	if (!m_AtomicBufferPrepared)
//		prepareAtomicCounterBuffer();
//
//	unsigned int *userCounters;
//	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCountersBuffer);
//	userCounters = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint) * (m_AtomicMaxID + 1),
//		GL_MAP_WRITE_BIT |
//		GL_MAP_INVALIDATE_BUFFER_BIT |
//		GL_MAP_UNSYNCHRONIZED_BIT);
//
//	memset(userCounters, 0, sizeof(GLuint) * (m_AtomicMaxID+1));
//
//	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
//	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
//}

//void
//GLRenderer::readAtomicCounters() {
//
//	unsigned int *userCounters;
//	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCountersBuffer);
//	userCounters = (GLuint *)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint) * (m_AtomicMaxID + 1),
//		GL_MAP_READ_BIT
//		);
//
//	memcpy(m_AtomicCounterValues, userCounters, sizeof(GLuint) * (m_AtomicMaxID + 1));
//
//	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
//	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
//}


#endif


// -----------------------------------------------------------------
//		LIGHTS
// -----------------------------------------------------------------

int
GLRenderer::getLightCount() {

	return m_Lights.size();
}


Light *
GLRenderer::getLight(unsigned int i) {

	assert(i < m_Lights.size());
	return m_Lights[i];
}


bool
GLRenderer::addLight(nau::scene::Light& aLight) {

	int id = m_Lights.size();

	aLight.setPropi(Light::ID, id);
	m_Lights.push_back(&aLight);
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
	case GL_TRIANGLES:
		m_TriCounter += size / 3;
		break;
	case GL_TRIANGLE_STRIP:
	case GL_TRIANGLE_FAN:
		m_TriCounter += size - 1;
		break;
	case GL_LINES:
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


int
GLRenderer::getNumberOfPrimitives(MaterialGroup *m) {

	unsigned int indices = m->getIndexData().getIndexSize();
	unsigned int primitive = m->getParent().getRealDrawingPrimitive();

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
#if NAU_OPENGL_VERSION >= 400			
	case GL_PATCHES:
		return indices / m->getParent().getnumberOfVerticesPerPatch();
#endif
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
GLRenderer::setMaterial(float *diffuse, float *ambient, float *emission, float *specular, float shininess) {

	m_Material.setPropf(ColorMaterial::SHININESS, shininess);
	m_Material.setPropf4(ColorMaterial::DIFFUSE, diffuse[0], diffuse[1], diffuse[2], diffuse[3]);
	m_Material.setPropf4(ColorMaterial::AMBIENT, ambient[0], ambient[1], ambient[2], ambient[3]);
	m_Material.setPropf4(ColorMaterial::EMISSION, emission[0], emission[1], emission[2], emission[3]);
	m_Material.setPropf4(ColorMaterial::SPECULAR, specular[0], specular[1], specular[2], specular[3]);
}


void
GLRenderer::setMaterial(ColorMaterial &mat) {

	m_Material.clone(mat);
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
GLRenderer::getAttribLocation(std::string name) {

	return VertexData::getAttribIndex(name);
}


// -----------------------------------------------------------------
//		MATERIAL - IMAGE TEXTURE
// -----------------------------------------------------------------


#if NAU_OPENGL_VERSION >=  420

void
GLRenderer::addImageTexture(unsigned int aTexUnit, ImageTexture *t) {

	m_ImageTextures[aTexUnit] = t;
}


void
GLRenderer::removeImageTexture(unsigned int aTexUnit) {

	if (m_ImageTextures.count(aTexUnit))
		m_ImageTextures.erase(aTexUnit);
}


int
GLRenderer::getImageTextureCount() {

	return m_ImageTextures.size();
}


ImageTexture*
GLRenderer::getImageTexture(unsigned int aTexUnit) {

	if (m_ImageTextures.count(aTexUnit))
		return m_ImageTextures[aTexUnit];
	else
		return NULL;
}

#endif


// -----------------------------------------------------------------
//		MATERIAL - TEXTURE
// -----------------------------------------------------------------


void
GLRenderer::setActiveTextureUnit(unsigned int aTexUnit) {

	glActiveTexture(GL_TEXTURE0 + (int)aTexUnit);
}


void
GLRenderer::addTexture(MaterialTexture *t) {

	m_Textures[t->getPropi(MaterialTexture::UNIT)] = t;
	m_IntProps[TEXTURE_COUNT]++;
}


void
GLRenderer::removeTexture(unsigned int aTexUnit) {

	if (m_Textures.count(aTexUnit)) {
		m_Textures.erase(aTexUnit);
		m_IntProps[TEXTURE_COUNT]--;
	}
}


MaterialTexture *
GLRenderer::getMaterialTexture(int unit) {

	if (m_Textures.count(unit))
		return m_Textures[unit];
	else
		return NULL;
}


Texture *
GLRenderer::getTexture(int unit) {

	if (m_Textures.count(unit))
		return m_Textures[unit]->getTexture();
	else
		return NULL;
}


int
GLRenderer::getTextureCount() {

	return (int)m_IntProps[TEXTURE_COUNT];
}



// -----------------------------------------------------------------
//		FRAMEBUFFER OPS
// -----------------------------------------------------------------


void
GLRenderer::clearFrameBuffer(unsigned int b) {

	GLenum c = 0;

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
		glDepthMask(dm);
		m_glDefaultState.setPropb(IState::DEPTH_MASK, dm);
		m_glCurrState.setPropb(IState::DEPTH_MASK, dm);
		int df = translateStencilDepthFunc(p->getPrope(Pass::DEPTH_FUNC));
		glDepthFunc(df);
		m_glDefaultState.setPrope(IState::DEPTH_FUNC, df);
		m_glCurrState.setPrope(IState::DEPTH_FUNC, df);
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
		glClearStencil(p->getPropf(Pass::STENCIL_CLEAR_VALUE));
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

	glColorMask(r, g, b, a);
	bvec4 *bv = new bvec4(r, g, b, a);
	m_glCurrState.setPropb4(IState::COLOR_MASK_B4, *bv);
	m_glDefaultState.setPropb4(IState::COLOR_MASK_B4, *bv);
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
GLRenderer::drawGroup (MaterialGroup* aMatGroup) {

	IRenderable& aRenderable = aMatGroup->getParent();
	IndexData &indexData = aMatGroup->getIndexData();

	unsigned int drawPrimitive = aRenderable.getRealDrawingPrimitive();

#if NAU_OPENGL_VERSION >= 400
	if (drawPrimitive == GL_PATCHES) {
		int k = aRenderable.getnumberOfVerticesPerPatch();
		glPatchParameteri(GL_PATCH_VERTICES, k);
	}
#endif
	// this forces compilation for everything that is rendered!
	// required for animated objects
	if (!aMatGroup->isCompiled())
		aMatGroup->compile();

	unsigned int size;

	{
		PROFILE ("Bindings");

		aMatGroup->bind();
	}

	{		
		PROFILE_GL ("Draw elements");

		size = indexData.getIndexSize();

		if (size != 0) {
		
			glDrawElements(drawPrimitive, size, GL_UNSIGNED_INT, 0);
		}
		else {
			size = aRenderable.getVertexData().getNumberOfVertices();
			glDrawArrays(drawPrimitive, 0, size);
		}
	}

#ifdef PROFILE
	accumTriCounter(drawPrimitive, size);
#endif

	aMatGroup->unbind();
}


void
GLRenderer::setCullFace(Face aFace) {

	glCullFace(translateFace(aFace));
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


unsigned int
GLRenderer::translateStencilDepthFunc(int anOp) {

	unsigned int res;

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


unsigned int
GLRenderer::translateStencilOp(int aFunc) {

	switch (aFunc) {
	case Pass::KEEP:
		return(GL_KEEP);
	case Pass::ZERO:
		return (GL_ZERO);
	case Pass::REPLACE:
		return (GL_REPLACE);
	case Pass::INCR:
		return(GL_INCR);
	case Pass::INCR_WRAP:
		return(GL_INCR_WRAP);
	case Pass::DECR:
		return(GL_DECR);
	case Pass::DECR_WRAP:
		return(GL_DECR_WRAP);
	case Pass::INVERT:
		return(GL_INVERT);
	default:
		return(GL_KEEP);
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
		return GLPrimitiveTypes[aDrawPrimitive];
	else
		return GL_INVALID_ENUM;	
}

