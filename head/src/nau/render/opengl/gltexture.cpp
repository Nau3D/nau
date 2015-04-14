#include "nau/render/opengl/gltexture.h"

#include "nau.h"
#include "nau/render/irenderer.h"

#include <GL/glew.h>

using namespace nau::render;

std::map<unsigned int, GLTexture::TexDataTypes> GLTexture::TexDataType;
std::map<unsigned int, GLTexture::TexFormats> GLTexture::TexFormat;
std::map<unsigned int, GLTexture::TexIntFormats> GLTexture::TexIntFormat;

bool GLTexture::Inited = GLTexture::InitGL();

bool
GLTexture::InitGL() {

	TexFormat[GL_RED_INTEGER       ] = TexFormats("RED", 1);
	TexFormat[GL_RED               ] = TexFormats("RED",1);
	TexFormat[GL_RG                ] = TexFormats("RG", 2);
	TexFormat[GL_RGB               ] = TexFormats("RGB", 3);
	TexFormat[GL_RGBA              ] = TexFormats("RGBA",4);
	TexFormat[GL_DEPTH_COMPONENT ] = TexFormats("DEPTH_COMPONENT",1);
	TexFormat[GL_DEPTH_STENCIL ] = TexFormats("DEPTH32F_STENCIL8",2);

	TexDataType[GL_UNSIGNED_BYTE   ] = TexDataTypes("UNSIGNED_BYTE"  ,  8);
	TexDataType[GL_BYTE            ] = TexDataTypes("BYTE"           ,  8);
	TexDataType[GL_UNSIGNED_SHORT  ] = TexDataTypes("UNSIGNED_SHORT" , 16);
	TexDataType[GL_SHORT           ] = TexDataTypes("SHORT"          , 16);
	TexDataType[GL_UNSIGNED_INT    ] = TexDataTypes("UNSIGNED_INT"   , 32);
	TexDataType[GL_INT             ] = TexDataTypes("INT"            , 32);
	TexDataType[GL_FLOAT           ] = TexDataTypes("FLOAT"          , 32);
	TexDataType[GL_UNSIGNED_INT_8_8_8_8_REV       ] = TexDataTypes("UNSIGNED_INT_8_8_8_8_REV"       , 32);
	TexDataType[GL_UNSIGNED_INT_24_8              ] = TexDataTypes("UNSIGNED_INT_24_8"              , 32);
	TexDataType[GL_FLOAT_32_UNSIGNED_INT_24_8_REV ] = TexDataTypes("FLOAT_32_UNSIGNED_INT_24_8_REV" , 32);

	TexIntFormat[GL_R8                   ] = TexIntFormats("R8",   GL_RED, GL_UNSIGNED_BYTE);
	TexIntFormat[GL_R16                  ] = TexIntFormats("R16",  GL_RED, GL_UNSIGNED_SHORT);
	TexIntFormat[GL_R16F                 ] = TexIntFormats("R16F", GL_RED, GL_FLOAT);
	TexIntFormat[GL_R32F                 ] = TexIntFormats("R32F", GL_RED, GL_FLOAT);
	TexIntFormat[GL_R8I                  ] = TexIntFormats("R8I",  GL_RED_INTEGER, GL_BYTE);
	TexIntFormat[GL_R16I				 ] = TexIntFormats("R16I", GL_RED_INTEGER, GL_SHORT);
	TexIntFormat[GL_R32I                 ] = TexIntFormats("R32I", GL_RED_INTEGER, GL_INT);
	TexIntFormat[GL_R8UI                 ] = TexIntFormats("R8UI", GL_RED_INTEGER, GL_UNSIGNED_BYTE);
	TexIntFormat[GL_R16UI                ] = TexIntFormats("R16UI",GL_RED_INTEGER, GL_UNSIGNED_SHORT);
	TexIntFormat[GL_R32UI                ] = TexIntFormats("R32UI",GL_RED_INTEGER, GL_UNSIGNED_INT);
						     						 
	TexIntFormat[GL_RG8                  ] = TexIntFormats("RG8",   GL_RG, GL_UNSIGNED_BYTE);
	TexIntFormat[GL_RG16                 ] = TexIntFormats("RG16",  GL_RG, GL_UNSIGNED_SHORT);
	TexIntFormat[GL_RG16F                ] = TexIntFormats("RG16F", GL_RG, GL_FLOAT);
	TexIntFormat[GL_RG32F                ] = TexIntFormats("RG32F", GL_RG, GL_FLOAT);
	TexIntFormat[GL_RG8I                 ] = TexIntFormats("RG8I",  GL_RG_INTEGER, GL_BYTE);
	TexIntFormat[GL_RG16I                ] = TexIntFormats("RG16I", GL_RG_INTEGER, GL_SHORT);
	TexIntFormat[GL_RG32I                ] = TexIntFormats("RG32I", GL_RG_INTEGER, GL_INT);
	TexIntFormat[GL_RG8UI                ] = TexIntFormats("RG8UI", GL_RG_INTEGER, GL_UNSIGNED_BYTE);
	TexIntFormat[GL_RG16UI               ] = TexIntFormats("RG16UI",GL_RG_INTEGER, GL_UNSIGNED_SHORT);
	TexIntFormat[GL_RG32UI               ] = TexIntFormats("RG32UI",GL_RG_INTEGER, GL_UNSIGNED_INT);
													 
//	TexIntFormat[GL_RGBA                 ] = TexIntFormats("RGBA",    GL_RGBA, GL_UNSIGNED_BYTE);
	TexIntFormat[GL_RGBA8                ] = TexIntFormats("RGBA",   GL_RGBA, GL_UNSIGNED_BYTE);
	TexIntFormat[GL_RGBA16               ] = TexIntFormats("RGBA16",  GL_RGBA, GL_UNSIGNED_SHORT);
	TexIntFormat[GL_RGBA16F              ] = TexIntFormats("RGBA16F", GL_RGBA, GL_FLOAT);
	TexIntFormat[GL_RGBA32F              ] = TexIntFormats("RGBA32F", GL_RGBA, GL_FLOAT);
	TexIntFormat[GL_RGBA8I               ] = TexIntFormats("RGBA8I",  GL_RGBA_INTEGER, GL_BYTE);
	TexIntFormat[GL_RGBA16I              ] = TexIntFormats("RGBA16I", GL_RGBA_INTEGER, GL_SHORT);
	TexIntFormat[GL_RGBA32I              ] = TexIntFormats("RGBA32I", GL_RGBA_INTEGER, GL_INT);
	TexIntFormat[GL_RGBA8UI              ] = TexIntFormats("RGBA8UI", GL_RGBA_INTEGER, GL_UNSIGNED_BYTE);
	TexIntFormat[GL_RGBA16UI             ] = TexIntFormats("RGBA16UI",GL_RGBA_INTEGER, GL_UNSIGNED_SHORT);
	TexIntFormat[GL_RGBA32UI             ] = TexIntFormats("RGBA32UI",GL_RGBA_INTEGER, GL_UNSIGNED_INT);
														  
	TexIntFormat[GL_DEPTH_COMPONENT16    ] = TexIntFormats("DEPTH_COMPONENT16", GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT);
	TexIntFormat[GL_DEPTH_COMPONENT24    ] = TexIntFormats("DEPTH_COMPONENT24", GL_DEPTH_COMPONENT, GL_UNSIGNED_INT_24_8);
	TexIntFormat[GL_DEPTH_COMPONENT32F   ] = TexIntFormats("DEPTH_COMPONENT32F",GL_DEPTH_COMPONENT, GL_FLOAT);
	TexIntFormat[GL_DEPTH32F_STENCIL8    ] = TexIntFormats("DEPTH32F_STENCIL8", GL_DEPTH_STENCIL,GL_FLOAT_32_UNSIGNED_INT_24_8_REV);

	for (auto f:TexIntFormat) {
	
		Attribs.listAdd("INTERNAL_FORMAT", f.second.name,		f.first);
	}

	for (auto f:TexFormat) {

		Attribs.listAdd("FORMAT", f.second.name,		f.first);
	}

	for (auto f:TexDataType) {

		Attribs.listAdd("TYPE", f.second.name,		f.first);
	}

	Attribs.listAdd("DIMENSION", "TEXTURE_2D", GL_TEXTURE_2D);
	Attribs.listAdd("DIMENSION", "TEXTURE_3D", GL_TEXTURE_3D);
	Attribs.listAdd("DIMENSION", "TEXTURE_2D_MULTISAMPLE_ARRAY" , GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
	Attribs.listAdd("DIMENSION", "TEXTURE_2D_ARRAY" , GL_TEXTURE_2D_ARRAY);
	Attribs.listAdd("DIMENSION", "TEXTURE_2D_MULTISAMPLE" , GL_TEXTURE_2D_MULTISAMPLE);
	return(true);
};



int
GLTexture::GetCompatibleFormat(int dim, int internalFormat) {

	GLint result;

//#if NAU_OPENGL_VERSION >= 420
//	glGetInternalformativ(dim, internalFormat, GL_TEXTURE_IMAGE_FORMAT, 1, &result);
//#else
	result = TexIntFormat[internalFormat].format;
//#endif
	return result;
}


int 
GLTexture::GetCompatibleType(int dim, int internalFormat) {

	GLint result;

#if NAU_OPENGL_VERSION >= 420
	glGetInternalformativ(dim, internalFormat, GL_TEXTURE_IMAGE_TYPE, 1, &result);
#else
	result = TexIntFormat[internalFormat].type;
#endif
	return result;
}


int
GLTexture::GetNumberOfComponents(unsigned int format) {

	return(TexFormat[format].numComp);
}


int 
GLTexture::GetElementSize(unsigned int format, unsigned int type) {

	int nComp = GetNumberOfComponents(format);
	return nComp * TexDataType[type].bitDepth;
}


int
GLTexture::getNumberOfComponents(void) {

	return(TexFormat[m_EnumProps[FORMAT]].numComp);
}


int 
GLTexture::getElementSize() {

	int nComp = getNumberOfComponents();
	return nComp * TexDataType[m_EnumProps[TYPE]].bitDepth;
}


GLTexture::GLTexture(std::string label): Texture(label) {

}

	
GLTexture::GLTexture(std::string label, std::string anInternalFormat, int width, int height, int depth, int layers, int levels, int samples):
	Texture(label)//, "TEXTURE_2D", anInternalFormat, width, height)
{

	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = height;
	m_IntProps[DEPTH] = depth;
	m_IntProps[SAMPLES] = samples;
	m_IntProps[LEVELS] = levels;
	m_IntProps[LAYERS] = layers;
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
	//TexIntFormat[m_EnumProps[INTERNAL_FORMAT]].type;

	build();
//	m_EnumProps[DIMENSION] = GL_TEXTURE_2D;
//	if (levels > 0)
//		m_BoolProps[MIPMAP] = true;
//	else
//		m_BoolProps[MIPMAP] = false;
//	
//
//	m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
//	m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
//
//	glGenTextures(1, (GLuint *)&(m_IntProps[ID]));
//	glBindTexture (m_EnumProps[DIMENSION], m_IntProps[ID]);
//
//	//m_BoolProps[MIPMAP] = false;
//
//#if NAU_OPENGL_VERSION < 420 //|| NAU_OPTIX
//	glTexImage2D(m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT], m_IntProps[WIDTH], m_IntProps[HEIGHT], 0,
// 		m_EnumProps[FORMAT], m_EnumProps[TYPE], NULL);
//#else
//	glTexStorage2D(GL_TEXTURE_2D, levels, m_EnumProps[INTERNAL_FORMAT], width, height);
//#endif
//
//	m_IntProps[COMPONENT_COUNT] = getNumberOfComponents();
//	m_IntProps[ELEMENT_SIZE] = getElementSize();
//
//	glBindTexture (m_EnumProps[DIMENSION], 0);
}


GLTexture::GLTexture (std::string label, std::string anInternalFormat, std::string aFormat, 
		std::string aType, int width, int height, void* data, bool mipmap) :
	Texture (label)//, "TEXTURE_2D", anInternalFormat, aFormat, aType, width, height)
{
	m_EnumProps[DIMENSION] = GL_TEXTURE_2D;
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
	m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
	//m_EnumProps[TYPE] = Attribs.getListValueOp(TYPE, aType);
	m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
	//aType;//GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);

	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = height;
	m_IntProps[DEPTH] = 1;
	m_IntProps[SAMPLES] = 0;
	m_IntProps[LEVELS] = 0;

	if (data != NULL) {

		m_IntProps[COMPONENT_COUNT] = getNumberOfComponents();
		m_IntProps[ELEMENT_SIZE] = getElementSize();

		glGenTextures(1, (GLuint *)&(m_IntProps[ID]));
		glBindTexture(m_EnumProps[DIMENSION], m_IntProps[ID]);

		glTexImage2D(m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT], m_IntProps[WIDTH], m_IntProps[HEIGHT], 0,
			m_EnumProps[FORMAT], m_EnumProps[TYPE], data);

		m_BoolProps[MIPMAP] = mipmap;
		//#ifndef NAU_OPTIX
		if (mipmap)
			glGenerateMipmap(GL_TEXTURE_2D);
		//#else
		//		m_BoolProps[MIPMAP] = false;
		//#endif
		glBindTexture (m_EnumProps[DIMENSION], 0);
	}

	else {
		build();
	}
}


void 
GLTexture::build() {

	glGenTextures(1, (GLuint *)&(m_IntProps[ID]));

	// 2D Texture
	if (m_IntProps[HEIGHT] > 1 && m_IntProps[DEPTH] == 1) {

		// 2D Texture Array 
		if (m_IntProps[LAYERS] > 1) {

			// 2D Texture Array MultiSample
			if (m_IntProps[SAMPLES] > 1) {
				m_EnumProps[DIMENSION] = GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
				glBindTexture(m_EnumProps[DIMENSION], m_IntProps[ID]);
				m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
				m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]); 
#if NAU_OPENGL_VERSION < 420 

				glTexImage3DMultisample(m_EnumProps[DIMENSION], m_EnumProps[SAMPLES], m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS], false);
#else
				glTexStorage3DMultisample(m_EnumProps[DIMENSION], m_EnumProps[SAMPLES], m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS], false);
#endif
			}
			// 2D Texture Array 
			else {
				m_EnumProps[DIMENSION] = GL_TEXTURE_2D_ARRAY;
				glBindTexture(m_EnumProps[DIMENSION], m_IntProps[ID]);
				m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
				m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]); 
#if NAU_OPENGL_VERSION < 420 

				glTexImage3D(m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS],0,
					m_EnumProps[FORMAT], m_EnumProps[TYPE], NULL);
#else
				glTexStorage3D(m_EnumProps[DIMENSION], 1, m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS]);
#endif
			}
		}
		// regular 2D Texture
		else {
			// 2D Texture MultiSample
			if (m_IntProps[SAMPLES] > 1) {
				m_EnumProps[DIMENSION] = GL_TEXTURE_2D_MULTISAMPLE;
				glBindTexture(m_EnumProps[DIMENSION], m_IntProps[ID]);
				m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
				m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]); 
#if NAU_OPENGL_VERSION < 420 

				glTexImage2DMultisample(m_EnumProps[DIMENSION], m_IntProps[SAMPLES], m_EnumProps[INTERNAL_FORMAT], 
					m_IntProps[WIDTH], m_IntProps[HEIGHT], false);
#else
				glTexStorage2DMultisample(m_EnumProps[DIMENSION], m_IntProps[SAMPLES], m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], false);
#endif
			}
			// 2D Texture 
			else {
				m_EnumProps[DIMENSION] = GL_TEXTURE_2D;
				glBindTexture(m_EnumProps[DIMENSION], m_IntProps[ID]);
				m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
				m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]); 
#if NAU_OPENGL_VERSION < 420 

				glTexImage2D(m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], 0,
					m_EnumProps[FORMAT], m_EnumProps[TYPE], NULL);
#else
				glTexStorage2D(m_EnumProps[DIMENSION], 1, m_EnumProps[INTERNAL_FORMAT], m_IntProps[WIDTH], m_IntProps[HEIGHT]);
#endif

			}
		}
	}
	// 3D Texture
	else if (m_IntProps[HEIGHT] > 1 && m_IntProps[DEPTH] > 1) {
		m_EnumProps[DIMENSION] = GL_TEXTURE_3D;
		glBindTexture(m_EnumProps[DIMENSION], m_IntProps[ID]);
		m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
		m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]); 
#if NAU_OPENGL_VERSION < 420 

		glTexImage3D(m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT],
			m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[DEPTH], 0,
			m_EnumProps[FORMAT], m_EnumProps[TYPE], NULL);
#else
		glTexStorage3D(m_EnumProps[DIMENSION], 1, m_EnumProps[INTERNAL_FORMAT], m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[DEPTH]);
#endif

	}				
	//m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
	//m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
	m_IntProps[COMPONENT_COUNT] = getNumberOfComponents();
	m_IntProps[ELEMENT_SIZE] = getElementSize();

	m_BoolProps[MIPMAP] = false;
	glBindTexture(m_EnumProps[DIMENSION], 0);
}


GLTexture::~GLTexture(void)
{
	glDeleteTextures(1, (GLuint *)&(m_IntProps[ID]));
}


void 
GLTexture::prepare(unsigned int aUnit, TextureSampler *ts) {

	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(m_EnumProps[DIMENSION],m_IntProps[ID]);

	ts->prepare(aUnit, m_EnumProps[DIMENSION]);
}


void 
GLTexture::restore(unsigned int aUnit) {

	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(m_EnumProps[DIMENSION],0);

	GLTextureSampler::restore(aUnit, m_EnumProps[DIMENSION]);
}


void
GLTexture::clear() {

#if NAU_OPENGL_VERSION >= 440
	for (int i = 0; i < m_IntProps[LEVELS]; ++i)
		glClearTexImage(m_IntProps[ID], i, m_EnumProps[FORMAT], m_EnumProps[TYPE], NULL);
#endif
}


void
GLTexture::clearLevel(int l) {

#if NAU_OPENGL_VERSION >= 440
	if (l < m_IntProps[LEVELS])
		glClearTexImage(m_IntProps[ID], l, m_EnumProps[FORMAT], m_EnumProps[TYPE], NULL);
#endif
}


void 
GLTexture::generateMipmaps() {

	glBindTexture(m_EnumProps[DIMENSION], m_IntProps[ID]);
	glGenerateMipmap(m_EnumProps[DIMENSION]);
}