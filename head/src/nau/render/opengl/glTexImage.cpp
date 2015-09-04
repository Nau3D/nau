#include "nau/render/opengl/glTexImage.h"

using namespace nau::render;



GLTexImage::GLTexImage (ITexture *t) :
	ITexImage (t)
{
	m_DataType = t->getPrope(ITexture::TYPE);
	int len = m_Width * m_Height * m_Depth * m_NumComponents; 

	switch (m_DataType) {
		case GL_FLOAT: 
			m_DataSize = sizeof(float) * len;
			m_Data = (float *)malloc(sizeof(float) * len);
			break;
		case GL_UNSIGNED_INT:
			m_DataSize = sizeof(unsigned int) * len;
			m_Data = (unsigned int *)malloc(sizeof(unsigned int) * len);
			break;
		case GL_UNSIGNED_SHORT:
			m_DataSize = sizeof(unsigned short) * len;
			m_Data = (unsigned short *)malloc(sizeof(unsigned short) * len);
			break;
		case GL_UNSIGNED_BYTE:
		case GL_UNSIGNED_INT_8_8_8_8_REV:
			m_DataSize = sizeof(unsigned char) * len;
			m_Data = (unsigned char *)malloc(sizeof(unsigned char) * len);
			break;

		case GL_INT:
			m_DataSize = sizeof(int) * len;
			m_Data = (int *)malloc(sizeof(int) * len);
			break;
		case GL_BYTE:
			m_DataSize = sizeof(char) * len;
			m_Data = (char *)malloc(sizeof(char) * len);
			break;
		case GL_SHORT:
			m_DataSize = sizeof(short) * len;
			m_Data = (short *)malloc(sizeof(short) * len);
			break;

		default:
			assert(false && "Image data type not available in glteximage.cpp");
			m_Data = NULL;
	}
}


GLTexImage::~GLTexImage(void)
{
	if (m_Data)
		free(m_Data);
}



void
GLTexImage::update(void) {
	
	glFinish();
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
	unsigned int b;
	glGenBuffers(1, &b);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, b);
	glBufferData(GL_PIXEL_PACK_BUFFER, m_DataSize, NULL, GL_STREAM_READ);
	int texType = m_Texture->getPrope(ITexture::DIMENSION);
	glBindTexture(texType,m_Texture->getPropi(ITexture::ID));
	if (texType == GL_TEXTURE_CUBE_MAP)
		texType = GL_TEXTURE_CUBE_MAP_POSITIVE_X;
	glGetTexImage(texType, 0, m_Texture->getPrope(ITexture::FORMAT), m_DataType, 0);
	void *m_Data2 = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	memcpy(m_Data, m_Data2, m_DataSize);
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
	glDeleteBuffers(1, &b);
}



void *
GLTexImage::getData() 
{
	update();
	return m_Data;
}


unsigned char *
GLTexImage::getRGBData() {

	int texType = m_Texture->getPrope(ITexture::DIMENSION);

	if (texType == GL_TEXTURE_CUBE_MAP)
		texType = GL_TEXTURE_CUBE_MAP_POSITIVE_X;
	else if (texType != GL_TEXTURE_2D)
		return NULL;

	unsigned int dataSize = m_Width * m_Height * 3 * (sizeof(unsigned char));
	glFinish();
	glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
	unsigned int b;
	glGenBuffers(1, &b);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, b);
	glBufferData(GL_PIXEL_PACK_BUFFER, dataSize, NULL, GL_STREAM_READ);

	glBindTexture(texType,m_Texture->getPropi(ITexture::ID));

	unsigned char *data = (unsigned char *)malloc(dataSize);
	glGetTexImage(texType, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	void *data2 = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

	memcpy(data, data2, dataSize);
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
	glDeleteBuffers(1, &b);
	return data;
}
