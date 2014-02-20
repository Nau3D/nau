#include <nau/render/opengl/gltextureCubeMap.h>
#include <nau/render/opengl/gltexture.h>
#include <nau/math/mat4.h>

using namespace nau::render;

int GLTextureCubeMap::faces[6] = {
			GL_TEXTURE_CUBE_MAP_POSITIVE_X, 
			GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 
			GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 
			GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 
			GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 
			GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
			};


GLTextureCubeMap::GLTextureCubeMap (std::string label, std::vector<std::string> files, 
									std::string anInternalFormat, std::string aFormat, 
									std::string aType, int width, unsigned char** data, bool mipmap) :
	TextureCubeMap (label,files, anInternalFormat, aFormat, aType, width)
{

	m_EnumProps[FORMAT] = Attribs.getListValueOp(FORMAT, aFormat);
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
	m_EnumProps[TYPE] = Attribs.getListValueOp(TYPE, aType);

	glGenTextures (1, &(m_UIntProps[ID]));
	glBindTexture (GL_TEXTURE_CUBE_MAP, m_UIntProps[ID]);

	//glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	//glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	for(int i = 0 ; i < 6 ; i++) {
		
		glTexImage2D (GLTextureCubeMap::faces[i], 0, m_EnumProps[INTERNAL_FORMAT], 
						m_IntProps[WIDTH], m_IntProps[HEIGHT], 0, m_EnumProps[FORMAT], m_EnumProps[TYPE], data[i]);
	}
	m_BoolProps[MIPMAP] = mipmap;
	if (mipmap)
		glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

#if NAU_CORE_OPENGL == 0
	glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
	glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
	glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
#endif
	glBindTexture (GL_TEXTURE_CUBE_MAP, 0);
}


GLTextureCubeMap::~GLTextureCubeMap(void)
{
	glDeleteTextures (1, &(m_UIntProps[ID]));
}


int
GLTextureCubeMap::getNumberOfComponents(void) {

	switch(m_EnumProps[FORMAT]) {

		case GL_LUMINANCE:
		case GL_RED:
		case GL_DEPTH_COMPONENT:
			return 1;
		case GL_RG:
		case GL_DEPTH_STENCIL:
			return 2;
		case GL_RGB:
			return 3;
		case GL_RGBA:
			return 4;
		default:
			return 0;
	}
}


void 
GLTextureCubeMap::prepare(int aUnit, nau::material::TextureSampler *ts) {

	RENDERER->addTexture((IRenderer::TextureUnit)aUnit, this);
	IRenderer::TextureUnit tu = (IRenderer::TextureUnit)(IRenderer::TEXTURE_UNIT0+aUnit);
	//m_SamplerProps[UNIT] = aUnit;
	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(GL_TEXTURE_CUBE_MAP,m_UIntProps[ID]);
	glBindSampler(aUnit, ts->getPropui(TextureSampler::ID));

	ts->prepare(aUnit, GL_TEXTURE_CUBE_MAP);
}


void 
GLTextureCubeMap::restore(int aUnit) 
{
	IRenderer::TextureUnit tu = (IRenderer::TextureUnit)(IRenderer::TEXTURE_UNIT0+aUnit);

	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(GL_TEXTURE_CUBE_MAP,0);

#if NAU_OPENGL_VERSION > 320
	glBindSampler(aUnit, 0);
#endif
#if NAU_CORE_OPENGL == 0
	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);

	glDisable(GL_TEXTURE_GEN_S);
    glDisable(GL_TEXTURE_GEN_T);
    glDisable(GL_TEXTURE_GEN_R);

	glDisable(GL_TEXTURE_CUBE_MAP);
#endif
}



//void 
//GLTextureCubeMap::prepare(int aUnit, nau::render::IState *state) 
//{
//	IState::TextureUnit tu = (IState::TextureUnit)(IState::TEXTURE0 + aUnit);
//
//	glActiveTexture (GL_TEXTURE0+aUnit);
//	glBindTexture(GL_TEXTURE_CUBE_MAP,m_Id);
//#if NAU_OPENGL_VERSION < 400
//	glEnable(GL_TEXTURE_CUBE_MAP);
//	glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
//	glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
//	glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
//
//
//	nau::scene::Camera *theCam = RENDERMANAGER->getCurrentCamera();
//	glMatrixMode(GL_TEXTURE);
//	mat4 m;
//	m.copy(theCam->getPropm4(Camera::VIEW_INVERSE_MATRIX).getMatrix());
//	m.set(3,0,0.0);
//	m.set(3,1,0.0);
//	m.set(3,2,0.0);
//	glLoadMatrixf(m.getMatrix());
//	glMatrixMode(GL_MODELVIEW);
//
//	glEnable(GL_TEXTURE_GEN_S);
//    glEnable(GL_TEXTURE_GEN_T);
//    glEnable(GL_TEXTURE_GEN_R);
//#endif
//}
//
//void 
//GLTextureCubeMap::restore(int aUnit, nau::render::IState *state) 
//{
//	IState::TextureUnit tu = (IState::TextureUnit)(IState::TEXTURE0 + aUnit);
//
//	glActiveTexture (GL_TEXTURE0+aUnit);
//	glBindTexture(GL_TEXTURE_CUBE_MAP,0);
//#if NAU_OPENGL_VERSION < 400
//	glMatrixMode(GL_TEXTURE);
//	glLoadIdentity();
//	glMatrixMode(GL_MODELVIEW);
//
//	glDisable(GL_TEXTURE_GEN_S);
//    glDisable(GL_TEXTURE_GEN_T);
//    glDisable(GL_TEXTURE_GEN_R);
//
//	glDisable(GL_TEXTURE_CUBE_MAP);
//#endif
//}


