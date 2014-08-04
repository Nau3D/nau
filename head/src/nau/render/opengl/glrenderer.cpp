#include <nau/render/opengl/glrenderer.h>

#include <nau/debug/profile.h>

#include <nau/material/material.h> 
#include <nau/material/imaterialgroup.h>
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
	m_MatrixMode(IRenderer::MODEL_MATRIX),
	m_TexturingFlag(true),
	m_ActualRenderMode(IRenderer::MATERIAL_MODE),
	m_PrevRenderMode(IRenderer::MATERIAL_MODE),
	m_Shader (0)
{
	m_glCurrState.set();

	for (int i = 0; i < IRenderer::COUNT_MATRIXMODE ; i++)
		m_Matrices[i] = SimpleTransform();

	m_CurrentMatrix = &(m_Matrices[m_MatrixMode]);

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
GLRenderer::getNumberOfPrimitives(IMaterialGroup *m) {

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

void 
GLRenderer::prepareAtomicCounterBuffer() {

	m_AtomicCounterValues = (unsigned int *)malloc(sizeof(unsigned int)*(m_AtomicMaxID+1));
	glGenBuffers(1, &m_AtomicCountersBuffer);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCountersBuffer);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint) * (m_AtomicMaxID + 1), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, m_AtomicCountersBuffer);
	m_AtomicBufferPrepared = true;

}

void 
GLRenderer::resetAtomicCounters() {

	if (!m_AtomicBufferPrepared)
		prepareAtomicCounterBuffer();

	unsigned int *userCounters;
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCountersBuffer);
	userCounters = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint) * (m_AtomicMaxID + 1),
		GL_MAP_WRITE_BIT |
		GL_MAP_INVALIDATE_BUFFER_BIT |
		GL_MAP_UNSYNCHRONIZED_BIT);

	memset(userCounters, 0, sizeof(GLuint) * (m_AtomicMaxID+1));

	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

void
GLRenderer::readAtomicCounters() {

	unsigned int *userCounters;
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCountersBuffer);
	userCounters = (GLuint *)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint) * (m_AtomicMaxID + 1),
		GL_MAP_READ_BIT
		);

	memcpy(m_AtomicCounterValues, userCounters, sizeof(GLuint) * (m_AtomicMaxID + 1));

	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

#endif

unsigned int *
GLRenderer::getAtomicCounterValues() {

#if (NAU_OPENGL_VERSION >= 400)

	if (m_AtomicCount)
		readAtomicCounters();
#endif
	return m_AtomicCounterValues;

}
// =============== RENDER ===================

void
GLRenderer::drawGroup (IMaterialGroup* aMatGroup)
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
#if NAU_CORE_OPENGL == 1
	if (!indexData.isCompiled())
		indexData.compile(aRenderable.getVertexData());
#endif

	unsigned int size;

#if NAU_CORE_OPENGL == 0	
	// this is due to a menu option that allows us to disable geometry texturing
	if (!m_TexturingFlag) {
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_TEXTURE_1D);
		glDisable(GL_TEXTURE_3D);
		glDisable(GL_TEXTURE_CUBE_MAP);
	}
#endif
	{
		PROFILE ("Bindings");

#if NAU_CORE_OPENGL != 1
		if (!indexData.isCompiled())
			aRenderable.getVertexData().bind();
#endif
		// else we have VAOs
		indexData.bind();
	}

	{		
		PROFILE_GL ("Draw elements");

		size = indexData.getIndexSize();

#if NAU_CORE_OPENGL != 1
		std::vector<unsigned int>& indices = indexData.getIndexData();
#endif
		if (size != 0) {
		

#if NAU_CORE_OPENGL != 1
			if (indexData.isCompiled()) {
#endif
				glDrawElements(drawPrimitive, size, GL_UNSIGNED_INT, 0);
#if NAU_CORE_OPENGL != 1
			}
			else {
				glDrawElements (drawPrimitive, size, GL_UNSIGNED_INT, static_cast<unsigned int*>(&indices[0]));
			}
#endif
		}
		else {
			size = aRenderable.getVertexData().getNumberOfVertices();
			glDrawArrays(drawPrimitive, 0, size);
		}
	}

#ifdef PROFILE
	accumTriCounter(drawPrimitive, size);
#endif

#if NAU_CORE_OPENGL != 1
	if (!indexData.isCompiled())
		aRenderable.getVertexData().unbind();
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

#if (NAU_OPENGL_VERSION >= 400)
	if (m_AtomicCount)
		resetAtomicCounters();
#endif
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
GLRenderer::clear (unsigned int b)
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

	// scissor is enable when setting the viewport
	glDisable(GL_SCISSOR_TEST);
}



void 
GLRenderer::setProp(IRenderer::BoolProps aprop, bool value) {

	switch(aprop) {

		case IRenderer::DEPTH_CLAMPING:
			if (value)
				glEnable(GL_DEPTH_CLAMP);
			else
				glDisable(GL_DEPTH_CLAMP);
			break;
		case IRenderer::COLOR_ENABLE:
			break;
		case IRenderer::DEPTH_CLEAR:
			break;
		case IRenderer::DEPTH_ENABLE:
			if (value)
				glEnable(GL_DEPTH_TEST);
			else
				glDisable(GL_DEPTH_TEST);
			m_glDefaultState.setProp(IState::DEPTH_TEST, value);
			m_glCurrState.setProp(IState::DEPTH_TEST, value);
			break;
		case IRenderer::DEPTH_MASK:
			glDepthMask(value);
			m_glDefaultState.setProp(IState::DEPTH_MASK, value);
			m_glCurrState.setProp(IState::DEPTH_MASK, value);
			break;
		case IRenderer::STENCIL_CLEAR:
			break;
		case IRenderer::STENCIL_ENABLE:
			if (value)
				glEnable(GL_STENCIL_TEST);
			else
				glDisable(GL_STENCIL_TEST);
			break;
	}
}


void 
GLRenderer::setDepthFunc(int f) {

	glDepthFunc(f);
	m_glDefaultState.setProp(IState::DEPTH_FUNC, f);
	m_glCurrState.setProp(IState::DEPTH_FUNC, f);
}


void 
GLRenderer::setStencilFunc(StencilFunc f, int ref, unsigned int mask) {

	glStencilFunc(translate(f), ref, mask);
}


unsigned int 
GLRenderer::translate(StencilFunc aFunc) {

	unsigned int res;

	switch(aFunc){
	case IRenderer::NEVER:
		res = GL_NEVER;
		break;
	case IRenderer::ALWAYS:
		res = GL_ALWAYS;
		break;
	case IRenderer::LESS:
		res = GL_LESS;
		break;
	case IRenderer::LEQUAL:
		res = GL_LEQUAL;
		break;
	case IRenderer::EQUAL:
		res = GL_EQUAL;
		break;
	case IRenderer::GEQUAL:
		res = GL_GEQUAL;
		break;
	case IRenderer::GREATER:
		res = GL_GREATER;
		break;
	case IRenderer::NOT_EQUAL:
		res = GL_NOTEQUAL;
		break;
	}
	return res;

}


void 
GLRenderer::setStencilOp(StencilOp sfail, StencilOp dfail, StencilOp dpass) {

	glStencilOp(translate(sfail), translate(dfail), translate(dpass));
}


unsigned int 
GLRenderer::translate(StencilOp s) {

	switch( s) {
		case IRenderer::KEEP:
			return(GL_KEEP);
		case IRenderer::ZERO:
			return (GL_ZERO);
		case IRenderer::REPLACE:
			return (GL_REPLACE);
		case IRenderer::INCR:
			return(GL_INCR);
		case IRenderer::INCR_WRAP:
			return(GL_INCR_WRAP);
		case IRenderer::DECR:
			return(GL_DECR);
		case IRenderer::DECR_WRAP:
			return(GL_DECR_WRAP);
		case IRenderer::INVERT:
			return(GL_INVERT);
		default:
			return(GL_KEEP);
	}
}



void 
GLRenderer::setDepthClearValue(float v) {

	glClearDepth(v);
}





void 
GLRenderer::setStencilClearValue(int v) {

	glClearStencil(v);
}


void 
GLRenderer::setStencilMaskValue(int i) {

	glStencilMask(i);
}



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


void 
GLRenderer::setViewport(int width, int height) {

	glViewport (0,0,width,height);
}


void 
GLRenderer::setViewport(nau::render::Viewport *aViewport) {

	const vec2& vpOrigin = aViewport->getPropf2(Viewport::ORIGIN);
	const vec2& vpSize = aViewport->getPropf2(Viewport::SIZE);
	const vec4& vpColor = aViewport->getPropf4(Viewport::CLEAR_COLOR);

	glViewport ((int)vpOrigin.x, (int)vpOrigin.y, (int)vpSize.x, (int)vpSize.y);
	glClearColor (vpColor.x, vpColor.y, vpColor.z, vpColor.w);

	glEnable(GL_SCISSOR_TEST); // MARK - perform scissor test only if not using the whole window 
	glScissor((int)vpOrigin.x, (int)vpOrigin.y, (int)vpSize.x, (int)vpSize.y);
}


void 
GLRenderer::setCamera (nau::scene::Camera *aCamera)
{

	m_Camera = aCamera;
	setViewport(aCamera->getViewport());

	m_Matrices[IRenderer::PROJECTION_MATRIX].setMat44((mat4 &)(aCamera->getPropm4(Camera::PROJECTION_MATRIX)));
	m_Matrices[IRenderer::VIEW_MATRIX].setMat44((mat4 &)(aCamera->getPropm4(Camera::VIEW_MATRIX)));
	m_Matrices[IRenderer::MODEL_MATRIX].setIdentity();


#if NAU_CORE_OPENGL == 0

	const float *matp = aCamera->getPropm4(Camera::PROJECTION_MATRIX).getMatrix();
	const float *matv = aCamera->getPropm4(Camera::VIEW_MATRIX).getMatrix();

	glMatrixMode (GL_PROJECTION);
	glLoadMatrixf(matp);

	glMatrixMode (GL_MODELVIEW);
	glLoadMatrixf(matv);

#endif
}


Camera *
GLRenderer::getCamera() {

	return m_Camera;
}


// ==============  MATRICES  =================

void 
GLRenderer::loadIdentity (void)
{
#if NAU_CORE_OPENGL == 0

	glLoadIdentity();

#endif

	m_Matrices[m_MatrixMode].setIdentity();
}


void 
GLRenderer::setMatrixMode (MatrixMode mode)
{

#if NAU_CORE_OPENGL == 0

	GLenum glMode;

	switch (mode) {
		case PROJECTION_MATRIX:
			glMode = GL_PROJECTION;
			break;
		case VIEW_MATRIX:
		case MODEL_MATRIX:
			glMode = GL_MODELVIEW;
			break;
		case TEXTURE_MATRIX:
			glMode = GL_TEXTURE;
			break;
	}

	glMatrixMode (glMode);

#endif

	m_MatrixMode = mode;
	m_CurrentMatrix = &(m_Matrices[m_MatrixMode]);
}


void 
GLRenderer::applyTransform (const nau::math::ITransform &aTransform)
{
	m_Matrices[m_MatrixMode].compose(aTransform);

#if NAU_CORE_OPENGL == 0

	const float *mat = aTransform.getMat44().getMatrix();
	glMultMatrixf (mat);

#endif
}


void
GLRenderer::pushMatrix (void) 
{
	m_MatrixStack[m_MatrixMode].push_back(m_Matrices[m_MatrixMode]);

#if NAU_CORE_OPENGL == 0

	glPushMatrix();

#endif
}


void
GLRenderer::popMatrix (void) 
{
	m_Matrices[m_MatrixMode] = m_MatrixStack[m_MatrixMode].back();
	m_MatrixStack[m_MatrixMode].pop_back();

#if NAU_CORE_OPENGL == 0

	glPopMatrix();

#endif
}

void 
GLRenderer::translate (nau::math::vec3 &aVec)
{

	m_Matrices[m_MatrixMode].translate(aVec);

#if NAU_CORE_OPENGL == 0

	glTranslatef (aVec.x, aVec.y, aVec.z);

#endif
}

void 
GLRenderer::scale (nau::math::vec3 &aVec)
{
	m_Matrices[m_MatrixMode].scale(aVec);

#if NAU_CORE_OPENGL == 0

	glScalef (aVec.x, aVec.y, aVec.z);

#endif
}


void 
GLRenderer::rotate(float angle, nau::math::vec3 &axis)
{
	m_Matrices[m_MatrixMode].rotate(angle,axis);

#if NAU_CORE_OPENGL == 0

	glRotatef (angle, axis.x, axis.y, axis.z);

#endif
}


const float *
GLRenderer::getMatrix(IRenderer::MatrixType aMode) 
{

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


//float* 
//GLRenderer::getProjectionModelviewMatrix (void)
//{
//
//#if NAU_CORE_OPENGL == 0
//
//	float m[16], p[16];
//
//	glGetFloatv (GL_PROJECTION_MATRIX, p);
//	glGetFloatv (GL_MODELVIEW_MATRIX, m);
//
//	glPushMatrix();
//	glLoadMatrixf (p);
//	glMultMatrixf (m);
//	glGetFloatv (GL_MODELVIEW_MATRIX, m_ReturnMatrix);
//	glPopMatrix();
//
//#endif
//
//	return m_ReturnMatrix;
//}


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


void
GLRenderer::setMaterial( ColorMaterial &mat) 
{
	m_Material.clone(mat);
	//m_Material.setProp(ColorMaterial::SHININESS, mat.getPropf(ColorMaterial::SHININESS));//clone(mat);
	//m_Material.setProp(ColorMaterial::DIFFUSE, mat.getProp4f(ColorMaterial::DIFFUSE));//clone(mat);
	//m_Material.setProp(ColorMaterial::AMBIENT, mat.getProp4f(ColorMaterial::AMBIENT));//clone(mat);
	//m_Material.setProp(ColorMaterial::EMISSION, mat.getProp4f(ColorMaterial::EMISSION));//clone(mat);
	//m_Material.setProp(ColorMaterial::SPECULAR, mat.getProp4f(ColorMaterial::SPECULAR));//clone(mat);

#if NAU_CORE_OPENGL == 0
	glColor4fv(mat.getDiffuse());
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,  m_Material.getDiffuse());
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   m_Material.getAmbient());
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  m_Material.getSpecular());
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION,  m_Material.getEmission());
	glMaterialf(GL_FRONT_AND_BACK,  GL_SHININESS, m_Material.getShininess());
#endif
}


void
GLRenderer::setMaterial(float *diffuse, float *ambient, float *emission, float *specular, float shininess)
{
	m_Material.setProp(ColorMaterial::SHININESS, shininess);
	m_Material.setProp(ColorMaterial::DIFFUSE, diffuse);
	m_Material.setProp(ColorMaterial::AMBIENT, ambient);
	m_Material.setProp(ColorMaterial::EMISSION, emission);
	m_Material.setProp(ColorMaterial::SPECULAR, specular);
	//m_Material.setDiffuse(diffuse);
	//m_Material.setAmbient(ambient);
	//m_Material.setSpecular(specular);
	//m_Material.setEmission(emission);
	//m_Material.setShininess(shininess);

#if NAU_CORE_OPENGL == 0
	glColor4fv(diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emission);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
#endif
}


const vec4 &
GLRenderer::getColorProp4f(ColorMaterial::Float4Property prop) {

	return m_Material.getProp4f(prop);
}


float
GLRenderer::getColorPropf(ColorMaterial::FloatProperty prop) {

	return m_Material.getPropf(prop);
}




//const float *
//GLRenderer::getColor(ColorMaterial::ColorComponent aColor) 
//{
//	const float *f;
//
//	switch(aColor) {
//		case ColorMaterial::DIFFUSE: f = m_Material.getDiffuse(); break;
//		case ColorMaterial::AMBIENT: f = m_Material.getAmbient(); break;
//		case ColorMaterial::SPECULAR: f = m_Material.getSpecular(); break;
//		case ColorMaterial::EMISSION: f = m_Material.getEmission(); break;
//		case ColorMaterial::SHININESS: f = m_Material.getShininessPtr(); break;
//	}
//	
//	return f;
//	
//}


void 
GLRenderer::setColor (float r, float g, float b, float a)
{

	m_Material.setProp(ColorMaterial::DIFFUSE, r,g,b,a);

#if NAU_CORE_OPENGL == 0
	glColor4f (r, g, b, a);
#endif
}


void 
GLRenderer::setColor (int r, int g, int b, int a)
{
	float m = 1.0f/255.0f;

	m_Material.setProp(ColorMaterial::DIFFUSE, r*m,g*m,b*m,a*m);

#if NAU_CORE_OPENGL == 0
	glColor4f (r*m, g*m, b*m, a*m);
#endif
}


void
GLRenderer::setState (IState *aState)
{
	m_glCurrState.setDiff (&m_glDefaultState, aState);
}


void 
GLRenderer::setDefaultState()
{
	m_glCurrState.setDefault();
	m_glDefaultState.setDefault();
	m_glCurrState.set();
}


void 
GLRenderer::setCullFace (Face aFace)
{
	glCullFace (translateFace(aFace));
}


// =================   LIGHTS   ======================

int
GLRenderer::getLightCount() 
{
	return m_Lights.size();
}

Light *
GLRenderer::getLight(unsigned int i) {

	assert(i < m_Lights.size());
	return m_Lights[i];
}


bool 
GLRenderer::addLight (nau::scene::Light& aLight)
{
	int id = m_Lights.size();

#if NAU_CORE_OPENGL == 0
	if (id == 8) { /***MARK***/ //A Magic number :D
		return false;
	}
#endif

	aLight.setProp(Light::ID, id);
	m_Lights.push_back(&aLight);


#if NAU_CORE_OPENGL == 0

	if (m_LightsOn && aLight.getPropb(Light::ENABLED)) {

		GLuint lightID = GL_LIGHT0 + id;
		glLightfv (lightID, GL_AMBIENT, &(aLight.getPropfv(Light::AMBIENT).x));
		glLightfv (lightID, GL_DIFFUSE, &(aLight.getPropfv(Light::DIFFUSE).x));
		glLightfv (lightID, GL_SPECULAR, &(aLight.getPropfv(Light::SPECULAR).x));
		glEnable (lightID);
	}

#endif

	return true;
}

void 
GLRenderer::removeLights ()
{

	for (unsigned int i = 0; i < m_Lights.size(); i++) {
		m_Lights[i]->setProp(Light::ID, -1);
	}

	m_Lights.clear();

#if NAU_CORE_OPENGL == 0
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHT1);
	glDisable(GL_LIGHT2);
	glDisable(GL_LIGHT3);
	glDisable(GL_LIGHT4);
	glDisable(GL_LIGHT5);
	glDisable(GL_LIGHT6);
	glDisable(GL_LIGHT7);
#endif
}


#if NAU_CORE_OPENGL == 0

void 
GLRenderer::activateLighting (void)
{
	m_LightsOn = true;

	float amb[4] = {0.0,0.0,0.0,1.0};

	glEnable (GL_LIGHTING);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,amb);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,1);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,0);
}


void 
GLRenderer::deactivateLighting (void)
{
	m_LightsOn = false;

	glDisable (GL_LIGHTING);
}

void 
GLRenderer::positionLight (nau::scene::Light& aLight)
{

	vec4 glPosition;

	if (m_LightsOn) {
		int id = aLight.getPropi(Light::ID);


		int lightID = GL_LIGHT0 + id;
		
		if (Light::DIRECTIONAL == aLight.getPrope(Light::TYPE)) {
			vec4 direction = aLight.getPropfv(Light::DIRECTION);
			glPosition.set (-direction.x, -direction.y, -direction.z, 0.0f);
		} 
		else  {
			vec4 position = aLight.getPropfv(Light::POSITION);
			glPosition.set (position.x, position.y, position.z, 1.0f);
			glLightf(lightID, GL_CONSTANT_ATTENUATION, aLight.getPropf(Light::CONSTANT_ATT));
			glLightf(lightID, GL_LINEAR_ATTENUATION, aLight.getPropf(Light::LINEAR_ATT));
			glLightf(lightID, GL_QUADRATIC_ATTENUATION, aLight.getPropf(Light::QUADRATIC_ATT));
			if (Light::POSITIONAL == aLight.getPrope(Light::TYPE)) {
				glLightf(lightID, GL_SPOT_CUTOFF, 180.0f);
			}
			else {
				glLightfv(lightID, GL_SPOT_DIRECTION, &(aLight.getPropfv(Light::DIRECTION).x));
				glLightf(lightID, GL_SPOT_CUTOFF, aLight.getPropf(Light::SPOT_CUTOFF));
				glLightf(lightID, GL_SPOT_EXPONENT, aLight.getPropf(Light::SPOT_EXPONENT));
			}
		}

		if (-1 != id){
			glLightfv (GL_LIGHT0 + id, GL_POSITION, &(glPosition.x));
		}
	}
}

#endif

// =================   IMAGE TEXTURES   ======================

#if NAU_OPENGL_VERSION >=  420

void 
GLRenderer::addImageTexture(unsigned int aTexUnit, ImageTexture *t)
{
	if ((unsigned int)aTexUnit < m_ImageTextures.size())
		m_ImageTextures[aTexUnit] = t;
}


void 
GLRenderer::removeImageTexture(unsigned int aTexUnit)
{
	if ((unsigned int)aTexUnit < m_ImageTextures.size())
		m_ImageTextures[aTexUnit] = 0;
}


int
GLRenderer::getImageTextureCount()
{
	int count = 0;
	for (unsigned int i = 0 ; i < m_ImageTextures.size(); i++)
		if (m_ImageTextures[i] != 0)
			count++;

	return count;
}


ImageTexture*
GLRenderer::getImageTexture(unsigned int aTexUnit)
{
	if ((unsigned int)aTexUnit < m_ImageTextures.size())
		return m_ImageTextures[aTexUnit];
	else
		return NULL;
}

#endif
// =================   TEXTURES   ======================



void 
GLRenderer::addTexture(unsigned int aTexUnit, Texture *t)
{
	if ((unsigned int)aTexUnit < m_Textures.size())
		m_Textures[aTexUnit] = t;
}


void 
GLRenderer::removeTexture(unsigned int aTexUnit)
{
	if ((unsigned int)aTexUnit < m_Textures.size())
		m_Textures[aTexUnit] = 0;
}

int
GLRenderer::getTextureCount()
{
	int count = 0;
	for (unsigned int i = 0 ; i < m_Textures.size(); i++)
		if (m_Textures[i] != 0)
			count++;

	return count;
}


int 
GLRenderer::getPropi(unsigned int aTexUnit, Texture::IntProperty prop)
{
	if ((unsigned int)aTexUnit < m_Textures.size() && m_Textures[aTexUnit] != 0)
		return (m_Textures[aTexUnit]->getPropi(prop));
	else
		return -1;
}

void 
GLRenderer::setActiveTextureUnit(unsigned int aTexUnit)
{
	glActiveTexture (GL_TEXTURE0 + (int)aTexUnit);
}





#if NAU_CORE_OPENGL != 1
void 
GLRenderer::enableTexturing (void)
{
	glEnable (GL_TEXTURE_2D);
}


void 
GLRenderer::disableTexturing (void)
{
	glDisable (GL_TEXTURE_2D);
	glBindTexture (GL_TEXTURE_2D, 0);
}




void 
GLRenderer::enableTextureCoordsGen (void)
{
	glEnable (GL_TEXTURE_GEN_S);
	glEnable (GL_TEXTURE_GEN_T);
}


void 
GLRenderer::disableTextureCoordsGen (void)
{
	glDisable (GL_TEXTURE_GEN_S);
	glDisable (GL_TEXTURE_GEN_T);
}

#endif


void 
GLRenderer::colorMask (bool r, bool g, bool b, bool a)
{
	glColorMask (r, g, b, a);
	m_glCurrState.setProp(IState::COLOR_MASK_B4, r, g, b, a);
	m_glDefaultState.setProp(IState::COLOR_MASK_B4, r, g, b, a);
}

void 
GLRenderer::activateUserClipPlane (ClipPlane aClipPlane)
{
	glEnable (GL_CLIP_PLANE0 + (int)aClipPlane);
}


void 
GLRenderer::setUserClipPlane (ClipPlane aClipPlane, double *plane)
{
	glClipPlane (GL_CLIP_PLANE0 + (int)aClipPlane, plane);
}


void 
GLRenderer::deactivateUserClipPlane (ClipPlane aClipPlane)
{
	glDisable (GL_CLIP_PLANE0 + (int)aClipPlane);
}



#if NAU_CORE_OPENGL == 0
void
GLRenderer::enableFog (void)
{
	float fogColor[4] = { 0.3f, 0.48f, 0.57f, 1.0f };

	glFogfv (GL_FOG_COLOR, fogColor);
	glFogi(GL_FOG_MODE, GL_EXP);

	glFogf(GL_FOG_DENSITY, 0.045f);

	glFogf(GL_FOG_START, 0.5f);
	glFogf(GL_FOG_END, 50.0f);

	glHint(GL_FOG_HINT, GL_DONT_CARE);

	glEnable (GL_FOG);
}

void 
GLRenderer::disableFog (void)
{
	glDisable (GL_FOG);
}

#endif

void 
GLRenderer::flush (void)
{
	glFinish();
}

nau::math::vec3 
GLRenderer::readpixel (int x, int y)
{
	GLubyte pixels[3];

	glReadPixels (x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	
	vec3 pixel (pixels[0], pixels[1], pixels[2]);

	return pixel;
}


float
GLRenderer::getDepthAtPoint(int x, int y) 
{
	float f;
	glReadPixels( x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &f);
	return f; 
}


GLenum 
GLRenderer::translateFace (Face aFace)
{
	switch (aFace) {
		case FRONT:
			return GL_FRONT;
			break;
		case FRONT_AND_BACK:
			return GL_FRONT_AND_BACK;
			break;
		case BACK:
			return GL_BACK;
			break;
	    default:
		  return GL_INVALID_ENUM;
	}
}

unsigned int 
GLRenderer::translateDrawingPrimitive (unsigned int aDrawPrimitive)
{
	if (IRenderer::PRIMITIVE_TYPE_COUNT > aDrawPrimitive)
		return GLPrimitiveTypes[aDrawPrimitive];
	else
		return GL_INVALID_ENUM;

	//switch (aDrawPrimitive) {
	//	case TRIANGLES:
	//		return GL_TRIANGLES;
	//		break;
	//	case TRIANGLE_STRIP:
	//		return GL_TRIANGLE_STRIP;
	//		break;
	//	case TRIANGLE_FAN:
	//		return GL_TRIANGLE_FAN;
	//		break;
	//	case LINES:
	//		return GL_LINES;
	//		break;
	//	case LINE_LOOP:
	//		return GL_LINE_LOOP;
	//		break;
	//    default:
	//	  return GL_INVALID_ENUM;
	
}

//GLenum 
//GLRenderer::translateMaterialComponent (ColorMaterial::ColorComponent aMaterialComponent)
//{
//	switch (aMaterialComponent) {
//		case ColorMaterial::AMBIENT:
//			return GL_AMBIENT;
//			break;
//		case ColorMaterial::DIFFUSE:
//			return GL_DIFFUSE;
//			break;
//		case ColorMaterial::AMBIENT_AND_DIFFUSE:
//			return GL_AMBIENT_AND_DIFFUSE;
//			break;
//		case ColorMaterial::SPECULAR:
//			return GL_SPECULAR;
//			break;
//		case ColorMaterial::EMISSION:
//			return GL_EMISSION;
//			break;
//		case ColorMaterial::SHININESS:
//			return GL_SHININESS;
//			break;
//	    default:
//		  return GL_INVALID_ENUM; 
//	}
//}

//void 
//GLRenderer::activateDefaultLight (void)
//{
//	glEnable (GL_LIGHT0);
//}
//
//void 
//GLRenderer::deactivateDefaultLight (void)
//{
//	glDisable (GL_LIGHT0);
//}



//void 
//GLRenderer::disableSurfaceShaders (void)
//{
//	m_SurfaceShaders = false;
//}
//
//void 
//GLRenderer::enableSurfaceShaders (void)
//{
//	m_SurfaceShaders = true;
//}


//void 
//GLRenderer::setDepthCompare()
//{
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL);
//	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_INTENSITY);
//}



//int
//GLRenderer::getNextAvailableLight (void)
//{
//	for (unsigned int i = 1; i < 8; i++) {
//		if (0 == (m_LightsOn & (1 << i))) {
//			m_LightsOn ^= (1 << i);
//			return i;
//		}
//	}
//	return (-1);
//}
//
//void
//GLRenderer::setLightFree (int n)
//{
//	glDisable(GL_LIGHT0+n);
//	m_LightsOn ^= (1 << n);
//}

//void 
//GLRenderer::drawElements (unsigned int size, std::vector<unsigned int>& indices, unsigned int aDrawPrimitive)
//{
//	unsigned int drawPrimitive = translateDrawingPrimitive(aDrawPrimitive);
//	switch (drawPrimitive) {
//		case GL_TRIANGLES:
//			m_TriCounter += size / 3;
//			break;
//		case GL_TRIANGLE_STRIP:
//		case GL_TRIANGLE_FAN:
//			m_TriCounter += size - 1;
//			break;
//		case GL_LINES:
//			m_TriCounter += size/2;
//			break;
//	}
//
////	m_TriCounter += size / 3;
//	if (indices.size() > 0) {
//		glDrawElements (drawPrimitive, size, GL_UNSIGNED_INT, static_cast<unsigned int*>(&indices[0]));
//	} else {
//		glDrawElements (drawPrimitive, size, GL_UNSIGNED_INT, 0);
//	}
//}

//void 
//GLRenderer::unproject (nau::render::IRenderable &aRenderable, nau::scene::Camera& aCamera)
//{
//	VertexData& vd = aRenderable.getVertexData();
//
//	std::vector<VertexData::Attr>& vv = vd.getAttributeDataOf (VertexData::CUSTOM_ATTRIBUTE_ARRAY0);
//	std::vector<VertexData::Attr>& vertices = vd.getDataOf (VertexData::VERTEX_ARRAY);
//
//	if (vv == VertexData::NoData) {
//		std::vector<VertexData::Attr>* newVector = new std::vector<VertexData::Attr> (4);
//
//		vd.setAttributeDataFor (VertexData::CUSTOM_ATTRIBUTE_ARRAY0, newVector, VertexData::NOLOC);
//	}
//
//	std::vector<VertexData::Attr>& viewVector = vd.getAttributeDataOf (VertexData::CUSTOM_ATTRIBUTE_ARRAY0);
//
//	Viewport& vp = aCamera.getViewport();
//
//	int viewport[4] = { static_cast<int>(vp.getOrigin().x), 
//                            static_cast<int>(vp.getOrigin().y), 
//                            static_cast<int>(vp.getSize().x), 
//                            static_cast<int>(vp.getSize().y) 
//                          };
//	
//	int points[4][2];
//
//	for (int i = 0; i < 4; i++) {
//		points[i][0] = static_cast<int>(vertices.at(i).x);
//		points[i][1] = static_cast<int>(vertices.at(i).y);
//	}
//
//
//	//int points[4][2] = { {vp.getOrigin().x, vp.getOrigin().y}, {vp.getOrigin().x, vp.getSize().y},
//	//							{vp.getSize().x, vp.getSize().y}, {vp.getSize().x, vp.getOrigin().y} };
//
//	// GCC sez: unused var
//	//float vec[3];
//	mat4 aMat;
//	for (int i = 0; i < 4; i++) {
//		/***MARK***/ //Replace gluFunction  with custom functioin
//		/***MARK***/ //I'm assuming only the quad, thus the -1
//		SimpleTransform t;
//
//		t.compose (aCamera.getViewMatrix());
//
//		t.compose (aCamera.getProjectionMatrix());
//
//		t.invert();
//
//		vec3 v1;
//
//		v1.x = 2.0f * (points[i][0] - viewport[0]) / viewport[2] - 1.0f;
//		v1.y = 2.0f * (points[i][1] - viewport[1]) / viewport[3] - 1.0f;
//		v1.z = 19.f;
//
//		t.getMat44().transform (v1);
//
//		v1 -= aCamera.getPosition();
//		v1.normalize();
//
//		viewVector.at (i).set (v1.x, v1.y, v1.z);
//		/***MARK***/ //It is necessary to multiply by the view matrix, but since 
//		             //in this case is only meant for the final stage where the matrix is the identity
//		             //I'll skip that, for now.
//	}
//
//}


//void 
//GLRenderer::renderObject (nau::scene::ISceneObject &aSceneObject, unsigned int buffers)
//{
//
//	IRenderable &aRenderable = aSceneObject.getRenderable();
//
//	doRender (&aSceneObject, buffers, -1);

	//	if (false == m_SurfaceShaders) {
	
//	} else {
		//std::vector<IMaterialGroup*> vMaterialGroups = aRenderable.getMaterialGroups();
		//std::vector<IMaterialGroup*>::iterator groupIter;

		//groupIter = vMaterialGroups.begin();
		//
		//for ( ; groupIter != vMaterialGroups.end(); groupIter++ ) {
		//	IMaterialGroup *aGroup = (*groupIter);
		//	VertexData &indexData = aGroup->getVertexData();

		//	indexData.bind();

		//	CMaterial* aMaterial = 0;

		//	//if (0 != m_MaterialProvider) {
		//	aMaterial = m_MaterialProvider->getMaterialForRender (aGroup->getMaterialId());
		//	//}
		//	if (true == aMaterial->m_enabled) {
		//		//m_RenderQueue.insert (std::pair<int, IRenderable*> (aMaterial->m_priority, &aRenderable));
		//		if (0 == m_RenderQueue.count (aMaterial->m_priority)){
		//			m_RenderQueue[aMaterial->m_priority] = new std::map <CMaterial*, std::vector<std::pair<IMaterialGroup*, ITransform*>>*>;
		//		}
		//		std::map<CMaterial*, std::vector<std::pair<IMaterialGroup*, ITransform*>>*> *materialMap = m_RenderQueue[aMaterial->m_priority];

		//		if (0 == materialMap->count (aMaterial)) {
		//			(*materialMap)[aMaterial] = new std::vector<std::pair<IMaterialGroup*, ITransform*>>;
		//		}
		//		std::vector<std::pair<IMaterialGroup*, ITransform*>> *matGroupVec = (*materialMap)[aMaterial];
		//		matGroupVec->push_back (std::pair<IMaterialGroup*, ITransform*>(aGroup, aSceneObject._getTransformPtr()));
		//	}
		//}
//	}
//}
//
//void
//GLRenderer::doRender (nau::scene::ISceneObject *aSceneObject, unsigned int buffers, int priority)
//{

	//IRenderable *aRenderable = aSceneObject->_getRenderablePtr();

	//GlState naturalState;

	//naturalState.m_glBlend = 0;
	//naturalState.m_glFog = 0;

	//VertexData &vertexData = aRenderable->getVertexData();

	//if (0 != buffers) {
	//	vertexData.bind (buffers);
	//} else {
	//	vertexData.bind();
	//}

	//push();
	//applyTransform (aSceneObject->getTransform());

	//std::vector<IMaterialGroup*> vMaterialGroups = aRenderable->getMaterialGroups();
	//std::vector<IMaterialGroup*>::iterator groupIter;

	//groupIter = vMaterialGroups.begin();

	//for ( ; groupIter != vMaterialGroups.end(); groupIter++ ) {
	//	IMaterialGroup *aGroup = (*groupIter);
	//	VertexData &indexData = aGroup->getVertexData();

	//	indexData.bind();

	//	CMaterial* aMaterial = 0;

	//	if (0 != m_MaterialProvider) {
	//		aMaterial = m_MaterialProvider->getMaterialForRender (aGroup->getMaterialId());
	//	}

	//	if (priority < 0 || priority == aMaterial->m_priority) { 

	//		if (true == m_SurfaceShaders) {
	//			//const float *dif = aMaterial->color.getDiffuse();

	//			//glColor3fv (dif);
	//			if (true == m_FixedFunction) {
	//				aMaterial->prepareFixed();
	//			} else {
	//				aMaterial->prepare();
	//				m_glCurrState.setDiff (aMaterial->OGLstate);
	//			}
	//			if (aMaterial->m_transparent) {
	//			}
	//		}

	//		if (0 != m_Shader) { /***MARK***/
	//			m_Shader->setValueOfUniformByNamef ("materialId", aMaterial->m_shaderid);
	//		}
	//		
	//		drawElements (indexData.getIndexSize(), indexData.getIndexData());

	//		if (true == m_SurfaceShaders) {
	//			if (true == m_FixedFunction) {
	//				aMaterial->restoreFixed();
	//			} else {
	//				aMaterial->restore();
	//				m_glCurrState.setDiff (naturalState);
	//			}
	//			if (aMaterial->m_transparent) {
	//				//glEnable (GL_DEPTH_TEST);
	//				//	glDisable (GL_ALPHA);
	//			}
	//		}
	//	}

	//	indexData.unbind();
	//}

	//pop();
	//vertexData.unbind();
//}
//struct MaterialGroupSort
//{
//	nau::scene::IScene *m_MaterialProvider;
//
//	MaterialGroupSort (nau::scene::IScene *materialProvider) : m_MaterialProvider(materialProvider) {}
//	bool operator()(IMaterialGroup*& a, IMaterialGroup*& b)
//	{
//		CMaterial *am = m_MaterialProvider->getMaterialForRender (a->getMaterialId());
//		CMaterial *bm = m_MaterialProvider->getMaterialForRender (b->getMaterialId());
//		
//		return (am->m_priority < bm->m_priority);
//	}
//};


//void 
//GLRenderer::setMaterialProvider (IScene *aScene)
//{
//	m_MaterialProvider = aScene;
//}
//void 
//GLRenderer::startRender (void)
//{

	//std::map<int, std::map<CMaterial*, std::vector<std::pair<IMaterialGroup*, ITransform*>>*>* >::iterator mapIter;

	//mapIter = m_RenderQueue.begin();

	//for ( ; mapIter != m_RenderQueue.end(); mapIter++) {
	//	std::map <CMaterial*, std::vector<std::pair<IMaterialGroup*, ITransform*>>*> *aMap;
	//	std::map <CMaterial*, std::vector<std::pair<IMaterialGroup*, ITransform*>>*>::iterator mapIter2;

	//	aMap = (*mapIter).second;

	//	mapIter2 = aMap->begin();
	//	for ( ; mapIter2 != aMap->end(); mapIter2++) {
	//		delete (*mapIter2).second;
	//	}

	//	delete aMap;
	//}

	//m_RenderQueue.clear(); /***MARK***/ //Possible memory leak
//}
//
//void 
//GLRenderer::finishRender (void)
//{
	//int priority = 0;
	//int size = m_RenderQueue.size();
	//int rendered = 0;

	//std::map <int, std::map<CMaterial*, std::vector<std::pair<IMaterialGroup*, ITransform*>>*>* >::iterator renderQueueIter;

	//renderQueueIter = m_RenderQueue.begin();

	//for (; renderQueueIter != m_RenderQueue.end(); renderQueueIter++) {
	//	std::map<CMaterial*, std::vector<std::pair<IMaterialGroup*, ITransform*>>*>::iterator materialMapIter;

	//	materialMapIter = (*renderQueueIter).second->begin();

	//	for (; materialMapIter != (*renderQueueIter).second->end(); materialMapIter++) {
	//		CMaterial *aMat = (*materialMapIter).first;
	//		
	//		if (m_FixedFunction) {
	//			aMat->prepareFixed();
	//			if (4 == aMat->m_shaderid) {
	//				glDisable (GL_LIGHTING);
	//			}
	//			if (1 == aMat->m_shaderid) {
	//				glActiveTexture (GL_TEXTURE0);
	//				glMatrixMode (GL_TEXTURE);

	//				float timeAgo = DegToRad (clock() * 6.0f / CLOCKS_PER_SEC);

	//				glTranslatef (timeAgo, timeAgo, 0.0f);
	//				glMatrixMode (GL_MODELVIEW);
	//			}
	//			if (2 == aMat->m_priority) {
	//				glEnable(GL_ALPHA_TEST);
	//				glAlphaFunc (GL_GREATER, 0.25f);
	//				glDisable (GL_CULL_FACE);
	//			}
	//		} else {
	//			aMat->prepare();
	//		}
	//		m_glCurrState.setDiff (aMat->OGLstate);
	//		
	//		std::vector<std::pair<IMaterialGroup*, ITransform*>>::iterator matGroupsIter;

	//		matGroupsIter = (*materialMapIter).second->begin();

	//		for (; matGroupsIter != (*materialMapIter).second->end(); matGroupsIter++) {
	//			push();
	//			applyTransform (*(*matGroupsIter).second);
	//			renderMaterialGroup ((*matGroupsIter).first);
	//			pop();
	//		}

	//		if (m_FixedFunction) {
	//			aMat->restoreFixed();
	//			if (4 == aMat->m_shaderid) {
	//				glEnable (GL_LIGHTING);
	//			}
	//			if (1 == aMat->m_shaderid) {
	//				glActiveTexture (GL_TEXTURE0);
	//				glMatrixMode (GL_TEXTURE);
	//				glLoadIdentity();
	//				glMatrixMode (GL_MODELVIEW);
	//			}
	//			if (2 == aMat->m_priority) {
	//				glEnable (GL_CULL_FACE);
	//				glDisable (GL_ALPHA_TEST);
	//			}

	//		} else {
	//			aMat->restore();
	//		}
	//	}
	//}
//}


//void 
//GLRenderer::renderBoundingVolume (const nau::geometry::IBoundingVolume* aBoundingVolume)
//{
//	vec3 bbMin (aBoundingVolume->getMin());
//	vec3 bbMax (aBoundingVolume->getMax());
//
//	glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_CURRENT_BIT);
//
//	glDisable (GL_LIGHTING);
//	glDisable (GL_CULL_FACE);
//	glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
//
//	glBegin (GL_QUADS);
//	//Front
//	glColor3f(1.0f, 0.0f, 0.0f);
//	glVertex3f(bbMin.x, bbMin.y, bbMax.z);
//	glVertex3f(bbMax.x, bbMin.y, bbMax.z);
//	glVertex3f(bbMax.x, bbMax.y, bbMax.z);
//	glVertex3f(bbMin.x, bbMax.y, bbMax.z);
//
//	//Back
//	glColor3f(0.0f, 1.0f, 0.0f);
//	glVertex3f(bbMin.x, bbMin.y, bbMin.z);
//	glVertex3f(bbMax.x, bbMin.y, bbMin.z);
//	glVertex3f(bbMax.x, bbMax.y, bbMin.z);
//	glVertex3f(bbMin.x, bbMax.y, bbMin.z);
//
//	//Top
//	glColor3f(0.0f, 0.0f, 1.0f);
//	glVertex3f(bbMax.x, bbMax.y, bbMin.z);
//	glVertex3f(bbMax.x, bbMax.y, bbMax.z);
//	glVertex3f(bbMin.x, bbMax.y, bbMax.z);
//	glVertex3f(bbMin.x, bbMax.y, bbMin.z);
//
//	//Bottom
//	glColor3f(0.0f, 1.0f, 1.0f);
//	glVertex3f(bbMax.x, bbMin.y, bbMin.z);
//	glVertex3f(bbMax.x, bbMin.y, bbMax.z);
//	glVertex3f(bbMin.x, bbMin.y, bbMax.z);
//	glVertex3f(bbMin.x, bbMin.y, bbMin.z);
//
//	//Left
//	glColor3f(1.0f, 1.0f, 0.0f);
//	glVertex3f(bbMin.x, bbMin.y, bbMax.z);
//	glVertex3f(bbMin.x, bbMax.y, bbMax.z);
//	glVertex3f(bbMin.x, bbMax.y, bbMin.z);
//	glVertex3f(bbMin.x, bbMin.y, bbMin.z);
//
//	//Right
//	glColor3f(1.0f, 0.0f, 1.0f);
//	glVertex3f(bbMax.x, bbMin.y, bbMax.z);
//	glVertex3f(bbMax.x, bbMax.y, bbMax.z);
//	glVertex3f(bbMax.x, bbMax.y, bbMin.z);
//	glVertex3f(bbMax.x, bbMin.y, bbMin.z);
//
//	glEnd ();
//
//	glPopAttrib();
//	//glPopAttrib();
//	//glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
//	//glEnable (GL_CULL_FACE);
//	//glEnable (GL_LIGHTING);
//}

//void 
//GLRenderer::enableStereo (void)
//{
//	m_Stereo = true;
//}
//
//void 
//GLRenderer::disableStereo (void)
//{
//	m_Stereo = false;
//}
//
//bool 
//GLRenderer::isStereo (void)
//{
//	return m_Stereo;
//}

//void
//GLRenderer::setFixedFunction (bool fixed)
//{
//	m_FixedFunction = fixed;
//}
//void 
//GLRenderer::setDepthMask(bool b) {
//
//	if (b)
//		glDepthMask(GL_TRUE);
//	else
//		glDepthMask(GL_FALSE);
//}

//void 
//GLRenderer::enableDepthTest (void)
//{
//	glEnable (GL_DEPTH_TEST);
//}
//
//
//void 
//GLRenderer::disableDepthTest (void)
//{
//	glDisable (GL_DEPTH_TEST);
//}



//void 
//GLRenderer::enableDepthClamping (void)
//{
//	glEnable (GL_DEPTH_CLAMP);
////	glEnable(GL_POLYGON_OFFSET_FILL);
////	glPolygonOffset(-1, -1);
//}
//
//
//void 
//GLRenderer::disableDepthClamping (void)
//{
//	glDisable (GL_DEPTH_CLAMP);
////	glDisable(GL_POLYGON_OFFSET_FILL);
//}
