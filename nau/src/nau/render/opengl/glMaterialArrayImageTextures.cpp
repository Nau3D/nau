#include "nau/render/opengl/glMaterialArrayImageTextures.h"

#include "nau.h"

#include <glbinding/gl/gl.h>
using namespace gl;


using namespace nau::render;

// Note: there is a duplicate of this map in GLTexture due to static initialization issues
std::map<GLenum, GLTexture::TexIntFormats> GLMaterialImageTextureArray::TexIntFormat = {
	{ GL_R8                  , GLTexture::TexIntFormats("R8",   GL_RED, GL_UNSIGNED_BYTE) },
	{ GL_R16                 , GLTexture::TexIntFormats("R16",  GL_RED, GL_UNSIGNED_SHORT) },
	{ GL_R16F                , GLTexture::TexIntFormats("R16F", GL_RED, GL_FLOAT) },
	{ GL_R32F                , GLTexture::TexIntFormats("R32F", GL_RED, GL_FLOAT) },
	{ GL_R8I                 , GLTexture::TexIntFormats("R8I",  GL_RED_INTEGER, GL_BYTE) },
	{ GL_R16I				 , GLTexture::TexIntFormats("R16I", GL_RED_INTEGER, GL_SHORT) },
	{ GL_R32I                , GLTexture::TexIntFormats("R32I", GL_RED_INTEGER, GL_INT) },
	{ GL_R8UI                , GLTexture::TexIntFormats("R8UI", GL_RED_INTEGER, GL_UNSIGNED_BYTE) },
	{ GL_R16UI               , GLTexture::TexIntFormats("R16UI",GL_RED_INTEGER, GL_UNSIGNED_SHORT) },
	{ GL_R32UI               , GLTexture::TexIntFormats("R32UI",GL_RED_INTEGER, GL_UNSIGNED_INT) },

	{ GL_RG8                 , GLTexture::TexIntFormats("RG8",   GL_RG, GL_UNSIGNED_BYTE) },
	{ GL_RG16                , GLTexture::TexIntFormats("RG16",  GL_RG, GL_UNSIGNED_SHORT) },
	{ GL_RG16F               , GLTexture::TexIntFormats("RG16F", GL_RG, GL_FLOAT) },
	{ GL_RG32F               , GLTexture::TexIntFormats("RG32F", GL_RG, GL_FLOAT) },
	{ GL_RG8I                , GLTexture::TexIntFormats("RG8I",  GL_RG_INTEGER, GL_BYTE) },
	{ GL_RG16I               , GLTexture::TexIntFormats("RG16I", GL_RG_INTEGER, GL_SHORT) },
	{ GL_RG32I               , GLTexture::TexIntFormats("RG32I", GL_RG_INTEGER, GL_INT) },
	{ GL_RG8UI               , GLTexture::TexIntFormats("RG8UI", GL_RG_INTEGER, GL_UNSIGNED_BYTE) },
	{ GL_RG16UI              , GLTexture::TexIntFormats("RG16UI",GL_RG_INTEGER, GL_UNSIGNED_SHORT) },
	{ GL_RG32UI              , GLTexture::TexIntFormats("RG32UI",GL_RG_INTEGER, GL_UNSIGNED_INT) },

	{ GL_RGBA8               , GLTexture::TexIntFormats("RGBA",   GL_RGBA, GL_UNSIGNED_BYTE) },
	{ GL_RGBA16              , GLTexture::TexIntFormats("RGBA16",  GL_RGBA, GL_UNSIGNED_SHORT) },
	{ GL_RGBA16F             , GLTexture::TexIntFormats("RGBA16F", GL_RGBA, GL_FLOAT) },
	{ GL_RGBA32F             , GLTexture::TexIntFormats("RGBA32F", GL_RGBA, GL_FLOAT) },
	{ GL_RGBA8I              , GLTexture::TexIntFormats("RGBA8I",  GL_RGBA_INTEGER, GL_BYTE) },
	{ GL_RGBA16I             , GLTexture::TexIntFormats("RGBA16I", GL_RGBA_INTEGER, GL_SHORT) },
	{ GL_RGBA32I             , GLTexture::TexIntFormats("RGBA32I", GL_RGBA_INTEGER, GL_INT) },
	{ GL_RGBA8UI             , GLTexture::TexIntFormats("RGBA8UI", GL_RGBA_INTEGER, GL_UNSIGNED_BYTE) },
	{ GL_RGBA16UI            , GLTexture::TexIntFormats("RGBA16UI",GL_RGBA_INTEGER, GL_UNSIGNED_SHORT) },
	{ GL_RGBA32UI            , GLTexture::TexIntFormats("RGBA32UI",GL_RGBA_INTEGER, GL_UNSIGNED_INT) },

	{ GL_DEPTH_COMPONENT16   , GLTexture::TexIntFormats("DEPTH_COMPONENT16", GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT) },
	{ GL_DEPTH_COMPONENT24   , GLTexture::TexIntFormats("DEPTH_COMPONENT24", GL_DEPTH_COMPONENT, GL_UNSIGNED_INT_24_8) },
	{ GL_DEPTH_COMPONENT32F  , GLTexture::TexIntFormats("DEPTH_COMPONENT32F",GL_DEPTH_COMPONENT, GL_FLOAT) },
	{ GL_DEPTH32F_STENCIL8   , GLTexture::TexIntFormats("DEPTH32F_STENCIL8", GL_DEPTH_STENCIL,GL_FLOAT_32_UNSIGNED_INT_24_8_REV) }
};

bool GLMaterialImageTextureArray::Inited = GLMaterialImageTextureArray::InitGL();

bool
GLMaterialImageTextureArray::InitGL() {

	Attribs.listAdd("ACCESS", "READ_ONLY", (int)GL_READ_ONLY);
	Attribs.listAdd("ACCESS", "WRITE_ONLY", (int)GL_WRITE_ONLY);
	Attribs.listAdd("ACCESS", "READ_WRITE", (int)GL_READ_WRITE);

	for (auto f: TexIntFormat) {
	
		Attribs.listAdd("INTERNAL_FORMAT", f.second.name,		(int)f.first);
	}
	return(true);
};


GLMaterialImageTextureArray::GLMaterialImageTextureArray() : MaterialArrayOfImageTextures() {

	m_EnumProps[ACCESS] = (int)GL_READ_WRITE;
	m_UIntProps[LEVEL] = 0;
}


GLMaterialImageTextureArray::~GLMaterialImageTextureArray(void) {

}


