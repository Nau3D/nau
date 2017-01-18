#include "nau/render/opengl/glTexture.h"

#include "nau.h"
#include "nau/render/iAPISupport.h"

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>


using namespace nau::render;

std::map<GLenum, GLTexture::TexDataTypes> GLTexture::TexDataType = {
	{ GL_UNSIGNED_BYTE                 , TexDataTypes("UNSIGNED_BYTE"                  ,  8) },
	{ GL_BYTE                          , TexDataTypes("BYTE"                           ,  8) },
	{ GL_UNSIGNED_SHORT                , TexDataTypes("UNSIGNED_SHORT"                 , 16) },
	{ GL_SHORT                         , TexDataTypes("SHORT"                          , 16) },
	{ GL_UNSIGNED_INT                  , TexDataTypes("UNSIGNED_INT"                   , 32) },
	{ GL_INT                           , TexDataTypes("INT"                            , 32) },
	{ GL_FLOAT                         , TexDataTypes("FLOAT"                          , 32) },
	{ GL_UNSIGNED_INT_8_8_8_8_REV      , TexDataTypes("UNSIGNED_INT_8_8_8_8_REV"       ,  8) },
	{ GL_UNSIGNED_INT_24_8             , TexDataTypes("UNSIGNED_INT_24_8"              , 32) },
	{ GL_FLOAT_32_UNSIGNED_INT_24_8_REV, TexDataTypes("FLOAT_32_UNSIGNED_INT_24_8_REV" , 32) },
};

std::map<GLenum, GLTexture::TexFormats> GLTexture::TexFormat = {
	{ GL_RED_INTEGER    , TexFormats("RED", 1) },
	{ GL_RED            , TexFormats("RED",1) },
	{ GL_RG             , TexFormats("RG", 2) },
	{ GL_RGB            , TexFormats("RGB", 3) },
	{ GL_RGBA           , TexFormats("RGBA",4) },
	{ GL_DEPTH_COMPONENT, TexFormats("DEPTH_COMPONENT",1) },
	{ GL_DEPTH_STENCIL  , TexFormats("DEPTH32F_STENCIL8",2) },
};

// Note: there is a duplicate of this map in GLImageTexture and GLArrayOfTextures due to static initialization issues
std::map<GLenum, GLTexture::TexIntFormats> GLTexture::TexIntFormat = {
	{ GL_R8                  , TexIntFormats("R8",   GL_RED, GL_UNSIGNED_BYTE) },
	{ GL_R16                 , TexIntFormats("R16",  GL_RED, GL_UNSIGNED_SHORT) },
	{ GL_R16F                , TexIntFormats("R16F", GL_RED, GL_FLOAT) },
	{ GL_R32F                , TexIntFormats("R32F", GL_RED, GL_FLOAT) },
	{ GL_R8I                 , TexIntFormats("R8I",  GL_RED_INTEGER, GL_BYTE) },
	{ GL_R16I				 , TexIntFormats("R16I", GL_RED_INTEGER, GL_SHORT) },
	{ GL_R32I                , TexIntFormats("R32I", GL_RED_INTEGER, GL_INT) },
	{ GL_R8UI                , TexIntFormats("R8UI", GL_RED_INTEGER, GL_UNSIGNED_BYTE) },
	{ GL_R16UI               , TexIntFormats("R16UI",GL_RED_INTEGER, GL_UNSIGNED_SHORT) },
	{ GL_R32UI               , TexIntFormats("R32UI",GL_RED_INTEGER, GL_UNSIGNED_INT) },

	{ GL_RG8                 , TexIntFormats("RG8",   GL_RG, GL_UNSIGNED_BYTE) },
	{ GL_RG16                , TexIntFormats("RG16",  GL_RG, GL_UNSIGNED_SHORT) },
	{ GL_RG16F               , TexIntFormats("RG16F", GL_RG, GL_FLOAT) },
	{ GL_RG32F               , TexIntFormats("RG32F", GL_RG, GL_FLOAT) },
	{ GL_RG8I                , TexIntFormats("RG8I",  GL_RG_INTEGER, GL_BYTE) },
	{ GL_RG16I               , TexIntFormats("RG16I", GL_RG_INTEGER, GL_SHORT) },
	{ GL_RG32I               , TexIntFormats("RG32I", GL_RG_INTEGER, GL_INT) },
	{ GL_RG8UI               , TexIntFormats("RG8UI", GL_RG_INTEGER, GL_UNSIGNED_BYTE) },
	{ GL_RG16UI              , TexIntFormats("RG16UI",GL_RG_INTEGER, GL_UNSIGNED_SHORT) },
	{ GL_RG32UI              , TexIntFormats("RG32UI",GL_RG_INTEGER, GL_UNSIGNED_INT) },

	{ GL_RGBA8               , TexIntFormats("RGBA",   GL_RGBA, GL_UNSIGNED_BYTE) },
	{ GL_RGBA16              , TexIntFormats("RGBA16",  GL_RGBA, GL_UNSIGNED_SHORT) },
	{ GL_RGBA16F             , TexIntFormats("RGBA16F", GL_RGBA, GL_FLOAT) },
	{ GL_RGBA32F             , TexIntFormats("RGBA32F", GL_RGBA, GL_FLOAT) },
	{ GL_RGBA8I              , TexIntFormats("RGBA8I",  GL_RGBA_INTEGER, GL_BYTE) },
	{ GL_RGBA16I             , TexIntFormats("RGBA16I", GL_RGBA_INTEGER, GL_SHORT) },
	{ GL_RGBA32I             , TexIntFormats("RGBA32I", GL_RGBA_INTEGER, GL_INT) },
	{ GL_RGBA8UI             , TexIntFormats("RGBA8UI", GL_RGBA_INTEGER, GL_UNSIGNED_BYTE) },
	{ GL_RGBA16UI            , TexIntFormats("RGBA16UI",GL_RGBA_INTEGER, GL_UNSIGNED_SHORT) },
	{ GL_RGBA32UI            , TexIntFormats("RGBA32UI",GL_RGBA_INTEGER, GL_UNSIGNED_INT) },

	{ GL_DEPTH_COMPONENT16   , TexIntFormats("DEPTH_COMPONENT16", GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT) },
	{ GL_DEPTH_COMPONENT24   , TexIntFormats("DEPTH_COMPONENT24", GL_DEPTH_COMPONENT, GL_UNSIGNED_INT_24_8) },
	{ GL_DEPTH_COMPONENT32F  , TexIntFormats("DEPTH_COMPONENT32F",GL_DEPTH_COMPONENT, GL_FLOAT) },
	{ GL_DEPTH32F_STENCIL8   , TexIntFormats("DEPTH32F_STENCIL8", GL_DEPTH_STENCIL,GL_FLOAT_32_UNSIGNED_INT_24_8_REV) }
};

std::map<GLenum, GLenum> GLTexture::TextureBound = {
	{ GL_TEXTURE_2D, GL_TEXTURE_BINDING_2D },
	{ GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BINDING_2D_ARRAY },
	{ GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_BINDING_2D_MULTISAMPLE },
	{ GL_TEXTURE_2D_MULTISAMPLE_ARRAY, GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY },
	{ GL_TEXTURE_3D, GL_TEXTURE_BINDING_3D },
	{ GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BINDING_CUBE_MAP }
};


bool GLTexture::Inited = GLTexture::InitGL();


bool
GLTexture::InitGL() {

	//TextureBound[GL_TEXTURE_2D] = GL_TEXTURE_BINDING_2D;
	//TextureBound[GL_TEXTURE_2D_ARRAY] = GL_TEXTURE_BINDING_2D_ARRAY;
	//TextureBound[GL_TEXTURE_2D_MULTISAMPLE] = GL_TEXTURE_BINDING_2D_MULTISAMPLE;
	//TextureBound[GL_TEXTURE_2D_MULTISAMPLE_ARRAY] = GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY;
	//TextureBound[GL_TEXTURE_3D] = GL_TEXTURE_BINDING_3D;
	//TextureBound[GL_TEXTURE_CUBE_MAP] = GL_TEXTURE_BINDING_CUBE_MAP;

	//TexFormat[GL_RED_INTEGER       ] = TexFormats("RED", 1);
	//TexFormat[GL_RED               ] = TexFormats("RED",1);
	//TexFormat[GL_RG                ] = TexFormats("RG", 2);
	//TexFormat[GL_RGB               ] = TexFormats("RGB", 3);
	//TexFormat[GL_RGBA              ] = TexFormats("RGBA",4);
	//TexFormat[GL_DEPTH_COMPONENT ] = TexFormats("DEPTH_COMPONENT",1);
	//TexFormat[GL_DEPTH_STENCIL ] = TexFormats("DEPTH32F_STENCIL8",2);

	//TexDataType[GL_UNSIGNED_BYTE   ] = TexDataTypes("UNSIGNED_BYTE"  ,  8);
	//TexDataType[GL_BYTE            ] = TexDataTypes("BYTE"           ,  8);
	//TexDataType[GL_UNSIGNED_SHORT  ] = TexDataTypes("UNSIGNED_SHORT" , 16);
	//TexDataType[GL_SHORT           ] = TexDataTypes("SHORT"          , 16);
	//TexDataType[GL_UNSIGNED_INT    ] = TexDataTypes("UNSIGNED_INT"   , 32);
	//TexDataType[GL_INT             ] = TexDataTypes("INT"            , 32);
	//TexDataType[GL_FLOAT           ] = TexDataTypes("FLOAT"          , 32);
	//TexDataType[GL_UNSIGNED_INT_8_8_8_8_REV       ] = TexDataTypes("UNSIGNED_INT_8_8_8_8_REV"       , 8);
	//TexDataType[GL_UNSIGNED_INT_24_8              ] = TexDataTypes("UNSIGNED_INT_24_8"              , 32);
	//TexDataType[GL_FLOAT_32_UNSIGNED_INT_24_8_REV ] = TexDataTypes("FLOAT_32_UNSIGNED_INT_24_8_REV" , 32);

//	TexIntFormat[GL_R8                   ] = TexIntFormats("R8",   GL_RED, GL_UNSIGNED_BYTE);
//	TexIntFormat[GL_R16                  ] = TexIntFormats("R16",  GL_RED, GL_UNSIGNED_SHORT);
//	TexIntFormat[GL_R16F                 ] = TexIntFormats("R16F", GL_RED, GL_FLOAT);
//	TexIntFormat[GL_R32F                 ] = TexIntFormats("R32F", GL_RED, GL_FLOAT);
//	TexIntFormat[GL_R8I                  ] = TexIntFormats("R8I",  GL_RED_INTEGER, GL_BYTE);
//	TexIntFormat[GL_R16I				 ] = TexIntFormats("R16I", GL_RED_INTEGER, GL_SHORT);
//	TexIntFormat[GL_R32I                 ] = TexIntFormats("R32I", GL_RED_INTEGER, GL_INT);
//	TexIntFormat[GL_R8UI                 ] = TexIntFormats("R8UI", GL_RED_INTEGER, GL_UNSIGNED_BYTE);
//	TexIntFormat[GL_R16UI                ] = TexIntFormats("R16UI",GL_RED_INTEGER, GL_UNSIGNED_SHORT);
//	TexIntFormat[GL_R32UI                ] = TexIntFormats("R32UI",GL_RED_INTEGER, GL_UNSIGNED_INT);
//						     						 
//	TexIntFormat[GL_RG8                  ] = TexIntFormats("RG8",   GL_RG, GL_UNSIGNED_BYTE);
//	TexIntFormat[GL_RG16                 ] = TexIntFormats("RG16",  GL_RG, GL_UNSIGNED_SHORT);
//	TexIntFormat[GL_RG16F                ] = TexIntFormats("RG16F", GL_RG, GL_FLOAT);
//	TexIntFormat[GL_RG32F                ] = TexIntFormats("RG32F", GL_RG, GL_FLOAT);
//	TexIntFormat[GL_RG8I                 ] = TexIntFormats("RG8I",  GL_RG_INTEGER, GL_BYTE);
//	TexIntFormat[GL_RG16I                ] = TexIntFormats("RG16I", GL_RG_INTEGER, GL_SHORT);
//	TexIntFormat[GL_RG32I                ] = TexIntFormats("RG32I", GL_RG_INTEGER, GL_INT);
//	TexIntFormat[GL_RG8UI                ] = TexIntFormats("RG8UI", GL_RG_INTEGER, GL_UNSIGNED_BYTE);
//	TexIntFormat[GL_RG16UI               ] = TexIntFormats("RG16UI",GL_RG_INTEGER, GL_UNSIGNED_SHORT);
//	TexIntFormat[GL_RG32UI               ] = TexIntFormats("RG32UI",GL_RG_INTEGER, GL_UNSIGNED_INT);
//													 
////	TexIntFormat[GL_RGBA                 ] = TexIntFormats("RGBA",    GL_RGBA, GL_UNSIGNED_BYTE);
//	TexIntFormat[GL_RGBA8                ] = TexIntFormats("RGBA",   GL_RGBA, GL_UNSIGNED_BYTE);
//	TexIntFormat[GL_RGBA16               ] = TexIntFormats("RGBA16",  GL_RGBA, GL_UNSIGNED_SHORT);
//	TexIntFormat[GL_RGBA16F              ] = TexIntFormats("RGBA16F", GL_RGBA, GL_FLOAT);
//	TexIntFormat[GL_RGBA32F              ] = TexIntFormats("RGBA32F", GL_RGBA, GL_FLOAT);
//	TexIntFormat[GL_RGBA8I               ] = TexIntFormats("RGBA8I",  GL_RGBA_INTEGER, GL_BYTE);
//	TexIntFormat[GL_RGBA16I              ] = TexIntFormats("RGBA16I", GL_RGBA_INTEGER, GL_SHORT);
//	TexIntFormat[GL_RGBA32I              ] = TexIntFormats("RGBA32I", GL_RGBA_INTEGER, GL_INT);
//	TexIntFormat[GL_RGBA8UI              ] = TexIntFormats("RGBA8UI", GL_RGBA_INTEGER, GL_UNSIGNED_BYTE);
//	TexIntFormat[GL_RGBA16UI             ] = TexIntFormats("RGBA16UI",GL_RGBA_INTEGER, GL_UNSIGNED_SHORT);
//	TexIntFormat[GL_RGBA32UI             ] = TexIntFormats("RGBA32UI",GL_RGBA_INTEGER, GL_UNSIGNED_INT);
//														  
//	TexIntFormat[GL_DEPTH_COMPONENT16    ] = TexIntFormats("DEPTH_COMPONENT16", GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT);
//	TexIntFormat[GL_DEPTH_COMPONENT24    ] = TexIntFormats("DEPTH_COMPONENT24", GL_DEPTH_COMPONENT, GL_UNSIGNED_INT_24_8);
//	TexIntFormat[GL_DEPTH_COMPONENT32F   ] = TexIntFormats("DEPTH_COMPONENT32F",GL_DEPTH_COMPONENT, GL_FLOAT);
//	TexIntFormat[GL_DEPTH32F_STENCIL8    ] = TexIntFormats("DEPTH32F_STENCIL8", GL_DEPTH_STENCIL,GL_FLOAT_32_UNSIGNED_INT_24_8_REV);

	for (auto f:TexIntFormat) {
	
		Attribs.listAdd("INTERNAL_FORMAT", f.second.name,		(int)f.first);
	}
	NauInt def;
	def = NauInt((int)GL_RGBA8);
	Attribs.setDefault("INTERNAL_FORMAT", def);

	for (auto f:TexFormat) {

		Attribs.listAdd("FORMAT", f.second.name,		(int)f.first);
	}
	def = NauInt((int)GL_RGBA);
	Attribs.setDefault("FORMAT", def);

	for (auto f:TexDataType) {

		Attribs.listAdd("TYPE", f.second.name,		(int)f.first);
	}
	def = NauInt((int)GL_UNSIGNED_BYTE);
	Attribs.setDefault("TYPE", def);

	Attribs.listAdd("DIMENSION", "TEXTURE_2D", (int)GL_TEXTURE_2D);
	Attribs.listAdd("DIMENSION", "TEXTURE_3D", (int)GL_TEXTURE_3D);
	Attribs.listAdd("DIMENSION", "TEXTURE_2D_MULTISAMPLE_ARRAY" , (int)GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
	Attribs.listAdd("DIMENSION", "TEXTURE_2D_ARRAY" , (int)GL_TEXTURE_2D_ARRAY);
	Attribs.listAdd("DIMENSION", "TEXTURE_2D_MULTISAMPLE" , (int)GL_TEXTURE_2D_MULTISAMPLE);
	def = NauInt((int)GL_TEXTURE_2D);
	Attribs.setDefault("DIMENSION", def);
	return(true);
};



int
GLTexture::GetCompatibleFormat(int dim, int internalFormat) {

	GLenum result;

	result = TexIntFormat[(GLenum)internalFormat].format;
	return (int)result;
}


int 
GLTexture::GetCompatibleType(int dim, int internalFormat) {

	GLenum result;

//#if NAU_OPENGL_VERSION >= 420
//	glGetInternalformativ(dim, internalFormat, GL_TEXTURE_IMAGE_TYPE, 1, &result);
//#else
	result = TexIntFormat[(GLenum)internalFormat].type;
//#endif
	return (int)result;
}


int
GLTexture::GetNumberOfComponents(unsigned int format) {

	return(TexFormat[(GLenum)format].numComp);
}


int 
GLTexture::GetElementSize(unsigned int format, unsigned int type) {

	int nComp = GetNumberOfComponents(format);
	return nComp * TexDataType[(GLenum)type].bitDepth;
}


int
GLTexture::getNumberOfComponents(void) {

	return (int)TexFormat[(GLenum)m_EnumProps[FORMAT]].numComp;
}


int 
GLTexture::getElementSize() {

	int nComp = getNumberOfComponents();
	return nComp * TexDataType[(GLenum)m_EnumProps[TYPE]].bitDepth;
}


GLTexture::GLTexture(std::string label): ITexture(label) {

}

	
GLTexture::GLTexture(std::string label, std::string anInternalFormat, int width, int height, int depth, int layers, int levels, int samples):
	ITexture(label)//, "TEXTURE_2D", anInternalFormat, width, height)
{

	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = height;
	m_IntProps[DEPTH] = depth;
	m_IntProps[SAMPLES] = samples;
	m_IntProps[LEVELS] = levels;
	m_IntProps[LAYERS] = layers;
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
	//TexIntFormat[m_EnumProps[INTERNAL_FORMAT]].type;
	if (m_IntProps[LEVELS] > 0)
		m_BoolProps[MIPMAP] = true;

	build();
}


GLTexture::GLTexture(std::string label, int anInternalFormat, int width, int height, int depth, int layers, int levels, int samples) :
	ITexture(label)//, "TEXTURE_2D", anInternalFormat, width, height)
{

	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = height;
	m_IntProps[DEPTH] = depth;
	m_IntProps[SAMPLES] = samples;
	m_IntProps[LEVELS] = levels;
	m_IntProps[LAYERS] = layers;
	m_EnumProps[INTERNAL_FORMAT] = anInternalFormat;
	//TexIntFormat[m_EnumProps[INTERNAL_FORMAT]].type;
	if (m_IntProps[LEVELS] > 0)
		m_BoolProps[MIPMAP] = true;

	build();
}


GLTexture::GLTexture (std::string label, std::string anInternalFormat, std::string aFormat,
		std::string aType, int width, int height, void* data, bool mipmap) :
	ITexture (label)//, "TEXTURE_2D", anInternalFormat, aFormat, aType, width, height)
{
	m_EnumProps[DIMENSION] = (int)GL_TEXTURE_2D;
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
		glBindTexture((GLenum)m_EnumProps[DIMENSION], m_IntProps[ID]);
		glTexImage2D((GLenum)m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT], m_IntProps[WIDTH], m_IntProps[HEIGHT], 0,
					(GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], data);

		m_BoolProps[MIPMAP] = mipmap;
		if (m_BoolProps[MIPMAP]) {
			generateMipmaps();
		}
		else {
			glTexParameteri((GLenum)m_EnumProps[DIMENSION], GL_TEXTURE_MAX_LEVEL, 0);
		}
		glBindTexture ((GLenum)m_EnumProps[DIMENSION], 0);
	}

	else {
		build();
	}
}

#include <algorithm>  

void 
GLTexture::build(int immutable) {

	int max;
	if (m_BoolProps[MIPMAP]) {
		max = std::max(m_IntProps[HEIGHT], std::max(m_IntProps[WIDTH], m_IntProps[DEPTH]));
		m_IntProps[LEVELS] = (int)log2(max);
	}
	else if (m_IntProps[LEVELS] >= 1)
		m_BoolProps[MIPMAP] = true;

	glGenTextures(1, (GLuint *)&(m_IntProps[ID]));
	
	// 2D ITexture
	if (m_IntProps[HEIGHT] > 1 && m_IntProps[DEPTH] == 1) {

		// 2D ITexture Array 
		if (m_IntProps[LAYERS] > 1) {

			// 2D ITexture Array MultiSample
			if (m_IntProps[SAMPLES] > 1) {
				m_EnumProps[DIMENSION] = (int)GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
				glBindTexture((GLenum)m_EnumProps[DIMENSION], m_IntProps[ID]);
				m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
				m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]); 
				if (immutable && APISupport->apiSupport(IAPISupport::TEX_STORAGE)) {
					glTexStorage3DMultisample((GLenum)m_EnumProps[DIMENSION], m_EnumProps[SAMPLES], (GLenum)m_EnumProps[INTERNAL_FORMAT],
						m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS], GL_FALSE);
				}
				else {
					glTexImage3DMultisample((GLenum)m_EnumProps[DIMENSION], m_EnumProps[SAMPLES], (GLenum)m_EnumProps[INTERNAL_FORMAT],
						m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS], GL_FALSE);
				}
			}
			// 2D ITexture Array 
			else {
				m_EnumProps[DIMENSION] = (int)GL_TEXTURE_2D_ARRAY;
				glBindTexture((GLenum)m_EnumProps[DIMENSION], m_IntProps[ID]);
				m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
				m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
				if (immutable && APISupport->apiSupport(IAPISupport::TEX_STORAGE)) {
					glTexStorage3D((GLenum)m_EnumProps[DIMENSION], m_IntProps[LEVELS], (GLenum)m_EnumProps[INTERNAL_FORMAT],
						m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS]);
				}
				else {
					glTexImage3D((GLenum)m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT],
						m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS], 0,
						(GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);
				}
			}
		}
		// regular 2D ITexture
		else {
			// 2D ITexture MultiSample
			if (m_IntProps[SAMPLES] > 1) {
				m_EnumProps[DIMENSION] = (int)GL_TEXTURE_2D_MULTISAMPLE;
				glBindTexture((GLenum)m_EnumProps[DIMENSION], m_IntProps[ID]);
				m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
				m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]); 
				if (immutable && APISupport->apiSupport(IAPISupport::TEX_STORAGE)) {
					glTexStorage2DMultisample((GLenum)m_EnumProps[DIMENSION], m_IntProps[SAMPLES], (GLenum)m_EnumProps[INTERNAL_FORMAT],
						m_IntProps[WIDTH], m_IntProps[HEIGHT], GL_FALSE);
				}
				else {
					glTexImage2DMultisample((GLenum)m_EnumProps[DIMENSION], m_IntProps[SAMPLES], (GLenum)m_EnumProps[INTERNAL_FORMAT],
						m_IntProps[WIDTH], m_IntProps[HEIGHT], GL_FALSE);
				}
			}
			// 2D ITexture 
			else {
				m_EnumProps[DIMENSION] = (int)GL_TEXTURE_2D;
				glBindTexture((GLenum)m_EnumProps[DIMENSION], m_IntProps[ID]);
				m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
				m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]); 
				if (immutable && APISupport->apiSupport(IAPISupport::TEX_STORAGE)) {
					glTexStorage2D((GLenum)m_EnumProps[DIMENSION], m_IntProps[LEVELS], (GLenum)m_EnumProps[INTERNAL_FORMAT], 
						m_IntProps[WIDTH], m_IntProps[HEIGHT]);
				}
				else {
					glTexImage2D((GLenum)m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT],
						m_IntProps[WIDTH], m_IntProps[HEIGHT], 0,
						(GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);
				}

			}
		}
	}
	// 3D ITexture
	else if (m_IntProps[HEIGHT] > 1 && m_IntProps[DEPTH] > 1) {
		m_EnumProps[DIMENSION] = (int)GL_TEXTURE_3D;
		glBindTexture((GLenum)m_EnumProps[DIMENSION], m_IntProps[ID]);
		m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
		m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]); 
		if (immutable && APISupport->apiSupport(IAPISupport::TEX_STORAGE)) {
			glTexStorage3D((GLenum)m_EnumProps[DIMENSION], m_IntProps[LEVELS], (GLenum)m_EnumProps[INTERNAL_FORMAT], 
				m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[DEPTH]);
		}
		else {
			glTexImage3D((GLenum)m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT],
				m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[DEPTH], 0,
				(GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);
		}
	}				
	//m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
	//m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
	m_IntProps[COMPONENT_COUNT] = getNumberOfComponents();
	m_IntProps[ELEMENT_SIZE] = getElementSize();

//	m_BoolProps[MIPMAP] = false;

	if (!immutable && m_IntProps[LEVELS] != 0) {//m_BoolProps[MIPMAP]) {
		glTexParameteriv((GLenum)m_EnumProps[DIMENSION], GL_TEXTURE_MAX_LEVEL, &m_IntProps[LEVELS]);
		glGenerateMipmap((GLenum)m_EnumProps[DIMENSION]);
	}
	else {
		glTexParameteri((GLenum)m_EnumProps[DIMENSION], GL_TEXTURE_MAX_LEVEL, m_IntProps[LEVELS]);
		//m_IntProps[LEVELS] = 0;
	}
	glBindTexture((GLenum)m_EnumProps[DIMENSION], 0);
}


void 
GLTexture::resize(unsigned int x, unsigned int y, unsigned int z) {

	m_IntProps[WIDTH] = x;
	m_IntProps[HEIGHT] = y;
	m_IntProps[DEPTH] = z;

	glBindTexture((GLenum)m_EnumProps[DIMENSION], m_IntProps[ID]);

	switch (m_EnumProps[DIMENSION]) {
	case (int)GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		glTexImage3DMultisample((GLenum)m_EnumProps[DIMENSION], m_EnumProps[SAMPLES], (GLenum)m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS], GL_FALSE);
		break;
	case (int)GL_TEXTURE_2D_ARRAY:
		glTexImage3D((GLenum)m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[LAYERS],0,
					(GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);
		break;
	case (int)GL_TEXTURE_3D:
		glTexImage3D((GLenum)m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[DEPTH],0,
					(GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);
		break;
	case (int)GL_TEXTURE_2D_MULTISAMPLE:
		glTexImage2DMultisample((GLenum)m_EnumProps[DIMENSION], m_IntProps[SAMPLES], (GLenum)m_EnumProps[INTERNAL_FORMAT], 
					m_IntProps[WIDTH], m_IntProps[HEIGHT], GL_FALSE);
		break;
	case (int)GL_TEXTURE_2D:
		glTexImage2D((GLenum)m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT],
					m_IntProps[WIDTH], m_IntProps[HEIGHT], 0,
					(GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);
		break;
	}
	//if (m_BoolProps[MIPMAP])
	//		glGenerateMipmap(m_EnumProps[DIMENSION]);
	//else
	//	glTexParameteri(m_EnumProps[DIMENSION], GL_TEXTURE_MAX_LEVEL, 0);
	if (m_BoolProps[MIPMAP]) {
		glGenerateMipmap((GLenum)m_EnumProps[DIMENSION]);
		glTexParameteriv((GLenum)m_EnumProps[DIMENSION], GL_TEXTURE_MAX_LEVEL, &m_IntProps[LEVELS]);
	}
	else {
		glTexParameteri((GLenum)m_EnumProps[DIMENSION], GL_TEXTURE_MAX_LEVEL, m_IntProps[LEVELS]);
		//m_IntProps[LEVELS] = 0;
	}

	glBindTexture((GLenum)m_EnumProps[DIMENSION], 0);
}


GLTexture::~GLTexture(void) {

	glDeleteTextures(1, (GLuint *)&(m_IntProps[ID]));
}


void 
GLTexture::prepare(unsigned int aUnit, ITextureSampler *ts) {

	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture((GLenum)m_EnumProps[DIMENSION],m_IntProps[ID]);

	ts->prepare(aUnit, m_EnumProps[DIMENSION]);
}


void 
GLTexture::restore(unsigned int aUnit, ITextureSampler *ts) {

	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture((GLenum)m_EnumProps[DIMENSION],0);

	ts->restore(aUnit, m_EnumProps[DIMENSION]);
}


void
GLTexture::clear() {

	IAPISupport *sup = IAPISupport::GetInstance();

	if (sup->apiSupport(IAPISupport::CLEAR_TEXTURE))
		for (int i = 0; i < m_IntProps[LEVELS]; ++i)
			glClearTexImage(m_IntProps[ID], i, (GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);

}


void
GLTexture::clearLevel(int l) {

	IAPISupport *sup = IAPISupport::GetInstance();

	if (sup->apiSupport(IAPISupport::CLEAR_TEXTURE_LEVEL) &&l < m_IntProps[LEVELS])
		glClearTexImage(m_IntProps[ID], l, (GLenum)m_EnumProps[FORMAT], (GLenum)m_EnumProps[TYPE], NULL);
}


void 
GLTexture::generateMipmaps() {

	glBindTexture((GLenum)m_EnumProps[DIMENSION], m_IntProps[ID]);
	m_BoolProps[MIPMAP] = true;
	int maxi = max(m_IntProps[WIDTH], max(m_IntProps[HEIGHT],m_IntProps[DEPTH]));
	m_IntProps[LEVELS] = (int)log2(maxi);
	glTexParameteri((GLenum)m_EnumProps[DIMENSION], GL_TEXTURE_MAX_LEVEL, m_IntProps[LEVELS]);
	glGenerateMipmap((GLenum)m_EnumProps[DIMENSION]);
//	glGetTexParameteriv((GLenum)m_EnumProps[DIMENSION], GL_TEXTURE_MAX_LEVEL, &m_IntProps[LEVELS]);
}