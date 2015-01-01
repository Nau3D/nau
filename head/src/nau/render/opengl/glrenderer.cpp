#include <nau/render/opengl/glrenderer.h>

#include <nau.h>
#include <nau/debug/profile.h>

#include <nau/material/material.h> 
#include <nau/material/materialgroup.h>
#include <nau/math/transformfactory.h>
#include <nau/render/opengl/glvertexarray.h>
#include <nau/render/opengl/glrendertarget.h>


#include <nau/slogger.h>

#include <nau/math/mat3.h>

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
	m_LightsOn (1),
	m_Lights(),
	m_Camera(NULL),
	m_Textures(),
	m_Matrices(IRenderer::COUNT_MATRIXMODE),
	m_TexturingFlag(true),
	m_ActualRenderMode(IRenderer::MATERIAL_MODE),
	m_PrevRenderMode(IRenderer::MATERIAL_MODE),
	m_Shader (0)
{
	m_glCurrState.set();

	for (int i = 0; i < IRenderer::COUNT_MATRIXMODE ; i++)
		m_Matrices[i] = SimpleTransform();

	m_CurrentMatrix = &(m_Matrices[MODEL_MATRIX]);

	//for (int i = 0 ; i < 8; i++)
	//	m_Textures.push_back(0);

	glEnable(GL_MULTISAMPLE);
}


GLRenderer::~GLRenderer(void)
{
	m_Lights.clear();
}


bool 
GLRenderer::init() 
{
	glewExperimental = true;
	GLenum error = glewInit();
	if (GLEW_OK != error){
		std::cout << "GLEW init error: " << glewGetErrorString(error) << std::endl;
		return false;
	}
	else {
		glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &MaxTextureUnits);
		for (int i = 0; i < MaxTextureUnits; ++i) {
			m_Textures.push_back(0);
#if NAU_OPENGL_VERSION >=  420
			m_ImageTextures.push_back(0);
#endif
		}
		glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &MaxColorAttachments);
		return true;
	}
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



// =============== ATOMIC COUNTERS ===================
#if (NAU_OPENGL_VERSION >= 400)

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


std::vector<unsigned int> &
GLRenderer::getAtomicCounterValues() {

	std::string buffer;
	unsigned int offset;
	unsigned int value;
	int i = 0;

	m_AtomicCounterValues.resize(m_AtomicLabels.size());

	if (m_AtomicCount) {
	
		for (auto at : m_AtomicLabels) {
			buffer = at.first.first;
			offset = at.first.second;
			IBuffer *b = RESOURCEMANAGER->getBuffer(buffer);
			if (NULL != b) {
				b->getData(offset, sizeof(unsigned int), &value);
				m_AtomicCounterValues[i++] = value;
			}
			else
				m_AtomicCounterValues[i++] = 0;
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	return m_AtomicCounterValues;

}

#endif

// =============== RENDER ===================

void
GLRenderer::drawGroup (MaterialGroup* aMatGroup)
{
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
	if (!indexData.isCompiled())
		indexData.compile(aRenderable.getVertexData());

	unsigned int size;

	{
		PROFILE ("Bindings");

		indexData.bind();
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

	aMatGroup->getIndexData().unbind();
}




// =============== PRIMITIVE COUNTER ===================


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
			m_TriCounter += size/2;
			break;
	}
}


void 
GLRenderer::resetCounters (void)
{
	m_TriCounter = 0;

//#if (NAU_OPENGL_VERSION >= 400)
//	if (m_AtomicCount)
//		resetAtomicCounters();
//#endif
}


unsigned int 
GLRenderer::getCounter (Counters c)
{
	if (c == TRIANGLE_COUNTER)
		return m_TriCounter;

	return 0;
}



// =============== SHADERS ===================

void 
GLRenderer::setShader (IProgram *aShader)
{
	m_Shader = aShader;
}


int
GLRenderer::getAttribLocation(std::string name) {

	return VertexData::getAttribIndex(name);
/*	if (m_Shader)
		return m_Shader->getAttributeLocation(name);
	else
		return -1;
*/}



// ================== ATTRIBUTES ====================
void 
GLRenderer::clearFrameBuffer(unsigned int b)
{
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

	glClear (c);

	// scissor is enabled when setting the viewport
	glDisable(GL_SCISSOR_TEST);
}


void 
GLRenderer::setDepthClamping(bool b) {

	if (b)
		glEnable(GL_DEPTH_CLAMP);
	else
		glDisable(GL_DEPTH_CLAMP);
}


void 
GLRenderer::prepareBuffers(Pass *p) {

	int clear = 0;

	bool value = p->getPropb(Pass::DEPTH_ENABLE);
	if (value) {
		glEnable(GL_DEPTH_TEST);
		bool dm = p->getPropb(Pass::DEPTH_MASK);
		glDepthMask(dm);
		m_glDefaultState.setProp(IState::DEPTH_MASK, dm);
		m_glCurrState.setProp(IState::DEPTH_MASK, dm);
		int df = translateStencilDepthFunc(p->getPrope(Pass::DEPTH_FUNC));
		glDepthFunc(df);
		m_glDefaultState.setProp(IState::DEPTH_FUNC, df);
		m_glCurrState.setProp(IState::DEPTH_FUNC, df);
	}
	else
		glDisable(GL_DEPTH_TEST);
	m_glDefaultState.setProp(IState::DEPTH_TEST, value);
	m_glCurrState.setProp(IState::DEPTH_TEST, value);

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

//void 
//GLRenderer::setPassProp(Pass::BoolProps aprop, bool value) {
//
//	switch(aprop) {
//
//		case IRenderer::DEPTH_CLAMPING:
//			if (value)
//				glEnable(GL_DEPTH_CLAMP);
//			else
//				glDisable(GL_DEPTH_CLAMP);
//			break;
//		case IRenderer::COLOR_ENABLE:
//			break;
//		case IRenderer::DEPTH_CLEAR:
//			break;
//		case IRenderer::DEPTH_ENABLE:
//			if (value)
//				glEnable(GL_DEPTH_TEST);
//			else
//				glDisable(GL_DEPTH_TEST);
//			m_glDefaultState.setProp(IState::DEPTH_TEST, value);
//			m_glCurrState.setProp(IState::DEPTH_TEST, value);
//			break;
//		case IRenderer::DEPTH_MASK:
//			glDepthMask(value);
//			m_glDefaultState.setProp(IState::DEPTH_MASK, value);
//			m_glCurrState.setProp(IState::DEPTH_MASK, value);
//			break;
//		case IRenderer::STENCIL_CLEAR:
//			break;
//		case IRenderer::STENCIL_ENABLE:
//			if (value)
//				glEnable(GL_STENCIL_TEST);
//			else
//				glDisable(GL_STENCIL_TEST);
//			break;
//	}
//}


//void 
//GLRenderer::setDepthFunc(int f) {
//
//	glDepthFunc(f);
//	m_glDefaultState.setProp(IState::DEPTH_FUNC, f);
//	m_glCurrState.setProp(IState::DEPTH_FUNC, f);
//}
//
//
//void 
//GLRenderer::setStencilFunc(StencilFunc f, int ref, unsigned int mask) {
//
//	glStencilFunc(translate(f), ref, mask);
//}


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


//void 
//GLRenderer::setStencilOp(StencilOp sfail, StencilOp dfail, StencilOp dpass) {
//
//	glStencilOp(translate(sfail), translate(dfail), translate(dpass));
//}


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



//void 
//GLRenderer::setDepthClearValue(float v) {
//
//	glClearDepth(v);
//}
//
//
//void 
//GLRenderer::setStencilClearValue(int v) {
//
//	glClearStencil(v);
//}
//
//
//void 
//GLRenderer::setStencilMaskValue(int i) {
//
//	glStencilMask(i);
//}


void 
GLRenderer::setRenderMode (TRenderMode mode)
{
	m_ActualRenderMode = mode;

	switch(mode) {
		case POINT_MODE:
			m_TexturingFlag = false;
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
			break;
		case WIREFRAME_MODE:
			m_TexturingFlag = false;
			glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
			break;
		case SOLID_MODE:
			m_TexturingFlag = false;
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			break;
		case MATERIAL_MODE:
			m_TexturingFlag = true;
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			break;
	}

}



Viewport *
GLRenderer::getViewport() {

	return m_Viewport;
}

//void 
//GLRenderer::setViewport(int width, int height) {
//
//	glViewport (0,0,width,height);
//}


void 
GLRenderer::setViewport(nau::render::Viewport *aViewport) {

	m_Viewport = aViewport;
	const vec2& vpOrigin = aViewport->getPropf2(Viewport::ABSOLUT_ORIGIN);
	const vec2& vpSize = aViewport->getPropf2(Viewport::ABSOLUT_SIZE);
	const vec4& vpColor = aViewport->getPropf4(Viewport::CLEAR_COLOR);

	glViewport ((int)vpOrigin.x, (int)vpOrigin.y, (int)vpSize.x, (int)vpSize.y);
	glClearColor (vpColor.x, vpColor.y, vpColor.z, vpColor.w);

	glEnable(GL_SCISSOR_TEST); // MARK - perform scissor test only if not using the whole window 
	glScissor((int)vpOrigin.x, (int)vpOrigin.y, (int)vpSize.x, (int)vpSize.y);
}

// ==============  CAMERA  =================

void 
GLRenderer::setCamera (nau::scene::Camera *aCamera) {

	m_Camera = aCamera;
	setViewport(aCamera->getViewport());

	m_Matrices[IRenderer::PROJECTION_MATRIX].setMat44((mat4 &)(aCamera->getPropm4(Camera::PROJECTION_MATRIX)));
	m_Matrices[IRenderer::VIEW_MATRIX].setMat44((mat4 &)(aCamera->getPropm4(Camera::VIEW_MATRIX)));
	m_Matrices[IRenderer::MODEL_MATRIX].setIdentity();
}


Camera *
GLRenderer::getCamera() {

	return m_Camera;
}


// ==============  MATRICES  =================
// See also setCamera

void 
GLRenderer::loadIdentity(MatrixMode mode) {

	m_Matrices[mode].setIdentity();
}


void 
GLRenderer::applyTransform(MatrixMode mode, const nau::math::ITransform &aTransform) {

	m_Matrices[mode].compose(aTransform);
}


void
GLRenderer::pushMatrix(MatrixMode mode) {

	m_MatrixStack[mode].push_back(m_Matrices[mode]);
}


void
GLRenderer::popMatrix(MatrixMode mode) {

	m_Matrices[mode] = m_MatrixStack[mode].back();
	m_MatrixStack[mode].pop_back();
}


void 
GLRenderer::translate(MatrixMode mode, nau::math::vec3 &aVec) {

	m_Matrices[mode].translate(aVec);
}


void 
GLRenderer::scale(MatrixMode mode, nau::math::vec3 &aVec) {

	m_Matrices[mode].scale(aVec);
}


void 
GLRenderer::rotate(MatrixMode mode, float angle, nau::math::vec3 &axis) {

	m_Matrices[mode].rotate(angle, axis);
}


const float *
GLRenderer::getMatrix(IRenderer::MatrixType aMode) {

	SimpleTransform s;

	// these matrices should be updated every time each of the components is updated
	// otherwise, light camera vectors will not work.
	switch (aMode) {
		case VIEW_MODEL:
			m_pReturnMatrix.copy(m_Matrices[IRenderer::VIEW_MATRIX].getMat44());
			m_pReturnMatrix.multiply(m_Matrices[IRenderer::MODEL_MATRIX].getMat44());
			break;
		case PROJECTION_VIEW_MODEL:
			m_pReturnMatrix.copy(m_Matrices[IRenderer::PROJECTION_MATRIX].getMat44());
			m_pReturnMatrix.multiply(m_Matrices[IRenderer::VIEW_MATRIX].getMat44());
			m_pReturnMatrix.multiply(m_Matrices[IRenderer::MODEL_MATRIX].getMat44());
			break;
		case PROJECTION_VIEW:
			m_pReturnMatrix.copy(m_Matrices[IRenderer::PROJECTION_MATRIX].getMat44());
			m_pReturnMatrix.multiply(m_Matrices[IRenderer::VIEW_MATRIX].getMat44());
			break;
		case TS05_PVM:
			s.setIdentity();
			s.translate (0.5f, 0.5f, 0.5f);
			s.scale (0.5f);	
			m_pReturnMatrix.copy(s.getMat44());
			m_pReturnMatrix.multiply(m_Matrices[IRenderer::PROJECTION_MATRIX].getMat44());
			m_pReturnMatrix.multiply(m_Matrices[IRenderer::VIEW_MATRIX].getMat44());
			break;
		case NORMAL:
			m_pReturnMatrix.copy(m_Matrices[IRenderer::VIEW_MATRIX].getMat44());
			m_pReturnMatrix.multiply(m_Matrices[IRenderer::MODEL_MATRIX].getMat44());
			m_pReturnMat3.setMatrix(m_pReturnMatrix.getSubMat3());
			m_pReturnMat3.invert();
			m_pReturnMat3.transpose();
			return m_pReturnMat3.getMatrix();
			break;

			// all other types
		default: return m_Matrices[aMode].getMat44().getMatrix();
	}
	return m_pReturnMatrix.getMatrix();
}


// ===========================================================


void
GLRenderer::saveAttrib(IRenderer::RendererAttributes aAttrib)
{
	switch(aAttrib) {
		case IRenderer::RENDER_MODE: 
			m_PrevRenderMode = m_ActualRenderMode;
			break;
	}
	
}


void
GLRenderer::restoreAttrib (void) 
{
	setRenderMode(m_PrevRenderMode);
}


// ================= MATERIAL ===================

void
GLRenderer::setMaterial(float *diffuse, float *ambient, float *emission, float *specular, float shininess) {

	m_Material.setProp(ColorMaterial::SHININESS, shininess);
	m_Material.setProp(ColorMaterial::DIFFUSE, diffuse);
	m_Material.setProp(ColorMaterial::AMBIENT, ambient);
	m_Material.setProp(ColorMaterial::EMISSION, emission);
	m_Material.setProp(ColorMaterial::SPECULAR, specular);
}


void
GLRenderer::setMaterial( ColorMaterial &mat) 
{
	m_Material.clone(mat);
}


const vec4 &
GLRenderer::getColorProp4f(ColorMaterial::Float4Property prop) {

	return m_Material.getPropf4(prop);
}


float
GLRenderer::getColorPropf(ColorMaterial::FloatProperty prop) {

	return m_Material.getPropf(prop);
}

float *
GLRenderer::getColorProp(int prop, Enums::DataType dt) {

	switch (dt) {

		case Enums::VEC4:
			m_vDummy = m_Material.getPropf4((ColorMaterial::Float4Property)prop);
			return &m_vDummy.x;
		case Enums::FLOAT:
			m_fDummy = m_Material.getPropf((ColorMaterial::FloatProperty)prop);
			return &m_fDummy;
	}
	return NULL;
}

void 
GLRenderer::setColor (float r, float g, float b, float a) {

	m_Material.setProp(ColorMaterial::DIFFUSE, r,g,b,a);
}


void 
GLRenderer::setColor (int r, int g, int b, int a) {

	float m = 1.0f/255.0f;

	m_Material.setProp(ColorMaterial::DIFFUSE, r*m,g*m,b*m,a*m);
}





// =============== STATE ==============================
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


void
GLRenderer::setState (IState *aState) {

	m_glCurrState.setDiff (&m_glDefaultState, aState);
}


void 
GLRenderer::setCullFace (Face aFace) {

	glCullFace (translateFace(aFace));
}


// =================   LIGHTS   ======================

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
GLRenderer::addLight (nau::scene::Light& aLight) {

	int id = m_Lights.size();

	aLight.setPropi(Light::ID, id);
	m_Lights.push_back(&aLight);

	return true;
}

void 
GLRenderer::removeLights () {

	for (unsigned int i = 0; i < m_Lights.size(); i++) {
		m_Lights[i]->setPropi(Light::ID, -1);
	}

	m_Lights.clear();
}




// =================   IMAGE TEXTURES   ======================

#if NAU_OPENGL_VERSION >=  420

void 
GLRenderer::addImageTexture(unsigned int aTexUnit, ImageTexture *t) {

	if ((unsigned int)aTexUnit < m_ImageTextures.size())
		m_ImageTextures[aTexUnit] = t;
}


void 
GLRenderer::removeImageTexture(unsigned int aTexUnit) {

	if ((unsigned int)aTexUnit < m_ImageTextures.size())
		m_ImageTextures[aTexUnit] = 0;
}


int
GLRenderer::getImageTextureCount() {

	int count = 0;
	for (unsigned int i = 0 ; i < m_ImageTextures.size(); i++)
		if (m_ImageTextures[i] != 0)
			count++;

	return count;
}


ImageTexture*
GLRenderer::getImageTexture(unsigned int aTexUnit) {

	if ((unsigned int)aTexUnit < m_ImageTextures.size())
		return m_ImageTextures[aTexUnit];
	else
		return NULL;
}

#endif


// =================   TEXTURES   ======================



void 
GLRenderer::addTexture(unsigned int aTexUnit, Texture *t) {

	if ((unsigned int)aTexUnit < m_Textures.size())
		m_Textures[aTexUnit] = t;
}


void 
GLRenderer::removeTexture(unsigned int aTexUnit) {

	if ((unsigned int)aTexUnit < m_Textures.size())
		m_Textures[aTexUnit] = 0;
}


int
GLRenderer::getTextureCount() {

	int count = 0;
	for (unsigned int i = 0 ; i < m_Textures.size(); i++)
		if (m_Textures[i] != 0)
			count++;

	return count;
}


Texture *
GLRenderer::getTexture(int unit){

	if ((unsigned int)unit < m_Textures.size())
		return m_Textures[unit];
	else
		return NULL;
}


int 
GLRenderer::getPropi(unsigned int aTexUnit, Texture::IntProperty prop) {

	if ((unsigned int)aTexUnit < m_Textures.size() && m_Textures[aTexUnit] != 0)
		return (m_Textures[aTexUnit]->getPropi(prop));
	else
		return -1;
}

void 
GLRenderer::setActiveTextureUnit(unsigned int aTexUnit) {

	glActiveTexture (GL_TEXTURE0 + (int)aTexUnit);
}



// =========== CLIP PLANES ===============

void 
GLRenderer::activateUserClipPlane(unsigned int  aClipPlane) {

	glEnable (GL_CLIP_PLANE0 + (int)aClipPlane);
}


void 
GLRenderer::setUserClipPlane(unsigned int  aClipPlane, double *plane) {

	glClipPlane (GL_CLIP_PLANE0 + (int)aClipPlane, plane);
}


void 
GLRenderer::deactivateUserClipPlane(unsigned int  aClipPlane) {

	glDisable (GL_CLIP_PLANE0 + (int)aClipPlane);
}



void 
GLRenderer::colorMask (bool r, bool g, bool b, bool a) {

	glColorMask (r, g, b, a);
	m_glCurrState.setProp(IState::COLOR_MASK_B4, r, g, b, a);
	m_glDefaultState.setProp(IState::COLOR_MASK_B4, r, g, b, a);
}
void 
GLRenderer::flush (void) {

	glFinish();
}


nau::math::vec3 
GLRenderer::readpixel (int x, int y) {

	GLubyte pixels[3];

	glReadPixels (x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	
	vec3 pixel (pixels[0], pixels[1], pixels[2]);

	return pixel;
}


float
GLRenderer::getDepthAtPoint(int x, int y) {

	float f;
	glReadPixels( x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &f);
	return f; 
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

