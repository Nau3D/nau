#include "nau/render/opengl/gltextureCubeMap.h"
#include "nau/render/opengl/gltexture.h"
#include "nau/math/matrix.h"

using namespace nau::render;

bool GLTextureCubeMap::Inited = GLTextureCubeMap::InitGL();

bool
GLTextureCubeMap::InitGL() {

	Attribs.listAdd("DIMENSION", "TEXTURE_CUBE_MAP", GL_TEXTURE_CUBE_MAP);
	return true;
}

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
	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = width;
	m_IntProps[DEPTH] = 1;
	m_IntProps[SAMPLES] = 0;
	m_IntProps[LEVELS] = 0;

	m_EnumProps[DIMENSION] = GL_TEXTURE_CUBE_MAP;
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
	m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
	m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);

	m_IntProps[COMPONENT_COUNT] = GLTexture::GetNumberOfComponents(m_EnumProps[FORMAT]);
	m_IntProps[ELEMENT_SIZE] = GLTexture::GetElementSize(m_EnumProps[FORMAT], m_EnumProps[TYPE]);

	glGenTextures(1, (GLuint *)&(m_IntProps[ID]));
	glBindTexture (GL_TEXTURE_CUBE_MAP, m_IntProps[ID]);

	for(int i = 0 ; i < 6 ; i++) {
		
		glTexImage2D (GLTextureCubeMap::faces[i], 0, m_EnumProps[INTERNAL_FORMAT], 
						m_IntProps[WIDTH], m_IntProps[HEIGHT], 0, m_EnumProps[FORMAT], m_EnumProps[TYPE], data[i]);
	}
	m_BoolProps[MIPMAP] = mipmap;
	if (mipmap)
		glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

	glBindTexture (GL_TEXTURE_CUBE_MAP, 0);
}


GLTextureCubeMap::~GLTextureCubeMap(void)
{
	glDeleteTextures(1, (GLuint *)&(m_IntProps[ID]));
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
GLTextureCubeMap::prepare(unsigned int aUnit, nau::material::TextureSampler *ts) {

	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(GL_TEXTURE_CUBE_MAP,m_IntProps[ID]);
	//glBindSampler(aUnit, ts->getPropi(TextureSampler::ID));

	ts->prepare(aUnit, GL_TEXTURE_CUBE_MAP);
}


void 
GLTextureCubeMap::restore(unsigned int aUnit) 
{
	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(GL_TEXTURE_CUBE_MAP,0);

#if NAU_OPENGL_VERSION > 320
	glBindSampler(aUnit, 0);
#endif
}


void 
GLTextureCubeMap::build() {}


void
GLTextureCubeMap::clear() {

#if NAU_OPENGL_VERSION >= 440
	for (int i = 0; i < m_IntProps[LEVELS]; ++i)
		glClearTexImage(m_UIntProps[ID], i, m_EnumProps[FORMAT], m_EnumProps[TYPE], NULL);
#endif
}


void
GLTextureCubeMap::clearLevel(int l) {

#if NAU_OPENGL_VERSION >= 440
	if (l < m_IntProps[LEVELS])
		glClearTexImage(m_UIntProps[ID], l, m_EnumProps[FORMAT], m_EnumProps[TYPE], NULL);
#endif
}


void 
GLTextureCubeMap::generateMipmaps() {

	glBindTexture(m_EnumProps[DIMENSION], m_UIntProps[ID]);
	glGenerateMipmap(m_EnumProps[DIMENSION]);
}

