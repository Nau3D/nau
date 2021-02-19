#include "nau/render/opengl/glArrayOfTextures.h"

#include "nau/slogger.h"

#include <glbinding/gl/gl.h>
using namespace gl;

// Note: there is a duplicate of this map in GLTexture and GLImageTexture due to static initialization issues
std::map<GLenum, GLArrayOfTextures::TexIntFormats> GLArrayOfTextures::TexIntFormat = {
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


bool GLArrayOfTextures::Inited = GLArrayOfTextures::InitGL();


bool
GLArrayOfTextures::InitGL() {

	for (auto f : TexIntFormat) {

		Attribs.listAdd("INTERNAL_FORMAT", f.second.name, (int)f.first);
	}

	return true;
}


GLArrayOfTextures::GLArrayOfTextures(const std::string &label): IArrayOfTextures(label) {

}


GLArrayOfTextures::~GLArrayOfTextures() {

}


void
GLArrayOfTextures::prepare(unsigned int firstUnit, ITextureSampler *ts) {

	if (m_Textures.size() == 0)
	return;

	const std::vector<int> &ids = m_IntArrayProps[TEXTURE_ID_ARRAY].getArray();
	int dim = m_Textures[0]->getPrope(ITexture::DIMENSION);

	for (int i = 0; i < (int)ids.size(); ++i) {
		glActiveTexture(GL_TEXTURE0 + i + firstUnit);
		glBindTexture((GLenum)dim, ids[i]);
		ts->prepare(i + firstUnit, m_Textures[i]->getPrope(ITexture::DIMENSION));
	}
}


void 
GLArrayOfTextures::restore(unsigned int firstUnit, ITextureSampler *ts) {

	if (m_Textures.size() == 0)
		return;

	int dim = m_Textures[0]->getPrope(ITexture::DIMENSION);

	for (int i = 0; i < (int)m_Textures.size(); ++i) {
		glActiveTexture(GL_TEXTURE0 + i + firstUnit);
		glBindTexture((GLenum)dim, 0);
		ts->restore(i + firstUnit, dim);
	}
}


void
GLArrayOfTextures::build() {

	for (unsigned int i = 0; i < m_UIntProps[TEXTURE_COUNT]; ++i) {

		std::string texLabel = m_Label + "_" + std::to_string(i);
		ITexture *tex = RESOURCEMANAGER->createTexture(texLabel, m_EnumProps[INTERNAL_FORMAT],
			m_IntProps[WIDTH], m_IntProps[HEIGHT], m_IntProps[DEPTH],
			m_IntProps[LAYERS], m_IntProps[LEVELS], m_IntProps[SAMPLES]);

		m_Textures.push_back(tex);
		m_IntArrayProps[TEXTURE_ID_ARRAY].append(tex->getPropi(ITexture::ID));

		tex->clear();
		int texID = tex->getPropi(ITexture::ID);
		if (APISupport->apiSupport(IAPISupport::APIFeatureSupport::BINDLESS_TEXTURES)) {
			uint64_t texPrt = glGetTextureHandleARB(texID);
			glMakeTextureHandleResidentARB(texPrt);
			m_TexturePointers.push_back(texPrt);
		}

	}

	// if buffer is defined
	if (APISupport->apiSupport(IAPISupport::APIFeatureSupport::BINDLESS_TEXTURES) && m_BoolProps[CREATE_BUFFER]) {

		if (RESOURCEMANAGER->hasBuffer(m_Label))
			SLOG("Warning: Buffer %s is defined more than once", m_Label.c_str());

		m_Buffer = RESOURCEMANAGER->createBuffer(m_Label);
		size_t bufferSize = m_TexturePointers.size() * sizeof(uint64_t);
		m_Buffer->setPropui(IBuffer::SIZE, (unsigned int)bufferSize);
		m_Buffer->setData(bufferSize, &m_TexturePointers[0]);
	}
}


void
GLArrayOfTextures::clearTextures() {

	for (unsigned int i = 0; i < m_UIntProps[TEXTURE_COUNT]; ++i) {
		m_Textures[i]->clear();
	}
}


void 
GLArrayOfTextures::clearTexturesLevel(int l) {

	for (unsigned int i = 0; i < m_UIntProps[TEXTURE_COUNT]; ++i) {
		m_Textures[i]->clearLevel(l);
	}
}


void
GLArrayOfTextures::generateMipmaps() {

	for (unsigned int i = 0; i < m_UIntProps[TEXTURE_COUNT]; ++i) {
		m_Textures[i]->generateMipmaps();
	}
}