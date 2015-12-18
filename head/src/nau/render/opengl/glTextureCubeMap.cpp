#include "nau/render/opengl/glTextureCubeMap.h"
#include "nau/render/opengl/glTexture.h"
#include "nau/math/matrix.h"


using namespace nau::render;

bool GLTextureCubeMap::Inited = GLTextureCubeMap::InitGL();

bool
GLTextureCubeMap::InitGL() {

	Attribs.listAdd("DIMENSION", "TEXTURE_CUBE_MAP", (int)GL_TEXTURE_CUBE_MAP);
	return true;
}

GLenum GLTextureCubeMap::faces[6] = {
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
	ITextureCubeMap (label,files, anInternalFormat, aFormat, aType, width)
{
	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = width;
	m_IntProps[DEPTH] = 1;
	m_IntProps[SAMPLES] = 0;
	m_IntProps[LEVELS] = 0;

	m_EnumProps[DIMENSION] = (int)GL_TEXTURE_CUBE_MAP;
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
	m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], 
										m_EnumProps[INTERNAL_FORMAT]);
	m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], 
										m_EnumProps[INTERNAL_FORMAT]);

	m_IntProps[COMPONENT_COUNT] = GLTexture::GetNumberOfComponents(m_EnumProps[FORMAT]);
	m_IntProps[ELEMENT_SIZE] = GLTexture::GetElementSize(m_EnumProps[FORMAT], m_EnumProps[TYPE]);

	glGenTextures(1, (GLuint *)&(m_IntProps[ID]));
	glBindTexture (GL_TEXTURE_CUBE_MAP, m_IntProps[ID]);

	for(int i = 0 ; i < 6 ; i++) {
		
		glTexImage2D (GLTextureCubeMap::faces[i], 0, m_EnumProps[INTERNAL_FORMAT], 
						m_IntProps[WIDTH], m_IntProps[HEIGHT], 0, 
						(GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], data[i]);
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

		case (unsigned int)GL_LUMINANCE:
		case (unsigned int)GL_RED:
		case (unsigned int)GL_DEPTH_COMPONENT:
			return 1;
		case (unsigned int)GL_RG:
		case (unsigned int)GL_DEPTH_STENCIL:
			return 2;
		case (unsigned int)GL_RGB:
			return 3;
		case (unsigned int)GL_RGBA:
			return 4;
		default:
			return 0;
	}
}


void 
GLTextureCubeMap::prepare(unsigned int aUnit, nau::material::ITextureSampler *ts) {

	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(GL_TEXTURE_CUBE_MAP,m_IntProps[ID]);
	//glBindSampler(aUnit, ts->getPropi(ITextureSampler::ID));

	ts->prepare(aUnit, (int)GL_TEXTURE_CUBE_MAP);
}


void 
GLTextureCubeMap::restore(unsigned int aUnit, nau::material::ITextureSampler *ts) 
{
	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(GL_TEXTURE_CUBE_MAP,0);


	ts->restore(aUnit, (int)GL_TEXTURE_CUBE_MAP);

}


void 
GLTextureCubeMap::build(int immutable) {}


void
GLTextureCubeMap::clear() {

	assert(APISupport->apiSupport(IAPISupport::CLEAR_TEXTURE) && "Clear Cubemap texture not supported");
	for (int i = 0; i < m_IntProps[LEVELS]; ++i)
		glClearTexImage(m_UIntProps[ID], i, (GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);
}


void
GLTextureCubeMap::clearLevel(int l) {

	assert(APISupport->apiSupport(IAPISupport::CLEAR_TEXTURE_LEVEL) && "Clear Cubemap texture level not supported");
	if (l < m_IntProps[LEVELS])
		glClearTexImage(m_UIntProps[ID], l, (GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);
}


void 
GLTextureCubeMap::generateMipmaps() {

	glBindTexture((GLenum)m_EnumProps[DIMENSION], m_UIntProps[ID]);
	glGenerateMipmap((GLenum)m_EnumProps[DIMENSION]);
}

