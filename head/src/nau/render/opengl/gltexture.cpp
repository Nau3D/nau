#include <nau/render/opengl/gltexture.h>
#include <nau.h>
#include <nau/render/irenderer.h>

using namespace nau::render;



bool GLTexture::Inited = GLTexture::InitGL();

bool
GLTexture::InitGL() {

	Attribs.listAdd("DIMENSION", "TEXTURE_1D", GL_TEXTURE_1D);
	Attribs.listAdd("DIMENSION", "TEXTURE_2D", GL_TEXTURE_2D);
	Attribs.listAdd("DIMENSION", "TEXTURE_3D", GL_TEXTURE_3D);
	Attribs.listAdd("DIMENSION", "TEXTURE_CUBE_MAP", GL_TEXTURE_CUBE_MAP);

	Attribs.listAdd("FORMAT", "LUMINANCE", GL_LUMINANCE);
	Attribs.listAdd("FORMAT", "RED", GL_RED);
	Attribs.listAdd("FORMAT", "RG", GL_RG);
	Attribs.listAdd("FORMAT", "RGB", GL_RGB);
	Attribs.listAdd("FORMAT", "RGBA", GL_RGBA);
	Attribs.listAdd("FORMAT", "DEPTH_COMPONENT", GL_DEPTH_COMPONENT);
	Attribs.listAdd("FORMAT", "DEPTH_STENCIL", GL_DEPTH_STENCIL);

	Attribs.listAdd("TYPE", "UNSIGNED_BYTE", GL_UNSIGNED_BYTE);
	Attribs.listAdd("TYPE", "BYTE", GL_BYTE);
	Attribs.listAdd("TYPE", "UNSIGNED_SHORT", GL_UNSIGNED_SHORT);
	Attribs.listAdd("TYPE", "SHORT", GL_SHORT);
	Attribs.listAdd("TYPE", "UNSIGNED_INT", GL_UNSIGNED_INT);
	Attribs.listAdd("TYPE", "INT", GL_INT);
	Attribs.listAdd("TYPE", "FLOAT", GL_FLOAT);
	Attribs.listAdd("TYPE", "HALF_FLOAT", GL_HALF_FLOAT);
	Attribs.listAdd("TYPE", "UNSIGNED_INT_24_8", GL_UNSIGNED_INT_24_8);

	Attribs.listAdd("INTERNAL_FORMAT", "R8",		GL_R8);
	Attribs.listAdd("INTERNAL_FORMAT", "R16",		GL_R16);
	Attribs.listAdd("INTERNAL_FORMAT", "RG8",		GL_RG8);
	Attribs.listAdd("INTERNAL_FORMAT", "RG16",		GL_RG16);
	Attribs.listAdd("INTERNAL_FORMAT", "RGB8",		GL_RGB8);
	Attribs.listAdd("INTERNAL_FORMAT", "RGB16",		GL_RGB16);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA8",		GL_RGBA8);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA16",	GL_RGBA16);
	Attribs.listAdd("INTERNAL_FORMAT", "R32F",		GL_R32F);
	Attribs.listAdd("INTERNAL_FORMAT", "RG16F",		GL_RG16F);
	Attribs.listAdd("INTERNAL_FORMAT", "RG32F",		GL_RG32F);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA",		GL_RGBA8); // For Optix
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA16F",	GL_RGBA16F);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA32F",	GL_RGBA32F);
	Attribs.listAdd("INTERNAL_FORMAT", "R32I",		GL_R32I);
	Attribs.listAdd("INTERNAL_FORMAT", "R32UI",		GL_R32UI);
	Attribs.listAdd("INTERNAL_FORMAT", "RG16I",		GL_RG16I);
	Attribs.listAdd("INTERNAL_FORMAT", "RG16UI",	GL_RG16UI);
	Attribs.listAdd("INTERNAL_FORMAT", "RG32I",		GL_RG32I);
	Attribs.listAdd("INTERNAL_FORMAT", "RG32UI",	GL_RG32UI);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA8I",	GL_RGBA8I);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA8UI",	GL_RGBA8UI);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA16I",	GL_RGBA16I);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA16UI",	GL_RGBA16UI);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA32I",	GL_RGBA32I);
	Attribs.listAdd("INTERNAL_FORMAT", "RGBA32UI",	GL_RGBA32UI);
	Attribs.listAdd("INTERNAL_FORMAT", "DEPTH_COMPONENT16", GL_DEPTH_COMPONENT16);
	Attribs.listAdd("INTERNAL_FORMAT", "DEPTH_COMPONENT24", GL_DEPTH_COMPONENT24);
	Attribs.listAdd("INTERNAL_FORMAT", "DEPTH_COMPONENT32", GL_DEPTH_COMPONENT32F);
	Attribs.listAdd("INTERNAL_FORMAT", "DEPTH24_STENCIL8",	GL_DEPTH24_STENCIL8);
	Attribs.listAdd("INTERNAL_FORMAT", "DEPTH32_STENCIL8",	GL_DEPTH32F_STENCIL8);

	return(true);
};


//#if NAU_OPENGL_VERSION < 420 || NAU_OPTIX
int
GLTexture::GetCompatibleFormat(int internalFormat) {

	switch(internalFormat) {
	
		case GL_DEPTH_COMPONENT16:
		case GL_DEPTH_COMPONENT24:
		case GL_DEPTH_COMPONENT32F:
			return GL_DEPTH_COMPONENT;
			
		case GL_DEPTH24_STENCIL8:
		case GL_DEPTH32F_STENCIL8:
			return GL_DEPTH_STENCIL;
		default:
			return GL_RGBA;
	}
}


int 
GLTexture::GetCompatibleType(int aFormat) {

	return GL_FLOAT;
}
//#endif


	
GLTexture::GLTexture(std::string label, std::string anInternalFormat, int width, int height, int levels):
	Texture(label, "TEXTURE_2D", anInternalFormat, width, height)
{
	m_EnumProps[DIMENSION] = GL_TEXTURE_2D;

	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = height;
	m_IntProps[DEPTH] = 1;
	m_IntProps[SAMPLES] = 0;
	m_IntProps[LEVELS] = levels;

	if (levels > 0)
		m_BoolProps[MIPMAP] = true;
	else
		m_BoolProps[MIPMAP] = false;
	
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);

#if NAU_OPENGL_VERSION < 420 || NAU_OPTIX
	m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[INTERNAL_FORMAT]);
	m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[FORMAT]);
#endif

	glGenTextures (1, &(m_UIntProps[ID]));
	glBindTexture (m_EnumProps[DIMENSION], m_UIntProps[ID]);

	m_BoolProps[MIPMAP] = false;

#if NAU_OPENGL_VERSION < 420 || NAU_OPTIX
	glTexImage2D(m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT], m_IntProps[WIDTH], m_IntProps[HEIGHT], 0,
 		m_EnumProps[FORMAT], m_EnumProps[TYPE], NULL);
#else
	glTexStorage2D(GL_TEXTURE_2D, levels, m_EnumProps[INTERNAL_FORMAT], width, height);
#endif

	m_IntProps[COMPONENT_COUNT] = getNumberOfComponents();
	m_IntProps[ELEMENT_SIZE] = getElementSize();

	glBindTexture (m_EnumProps[DIMENSION], 0);
}


GLTexture::GLTexture (std::string label, std::string anInternalFormat, std::string aFormat, 
		std::string aType, int width, int height, void* data, bool mipmap) :
	Texture (label, "TEXTURE_2D", anInternalFormat, aFormat, aType, width, height)
{
	m_EnumProps[DIMENSION] = GL_TEXTURE_2D;
	m_EnumProps[FORMAT] = Attribs.getListValueOp(FORMAT, aFormat);
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
	m_EnumProps[TYPE] = Attribs.getListValueOp(TYPE, aType);

	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = height;
	m_IntProps[DEPTH] = 1;
	m_IntProps[SAMPLES] = 0;
	m_IntProps[LEVELS] = 0;
	m_IntProps[COMPONENT_COUNT] = getNumberOfComponents();
	m_IntProps[ELEMENT_SIZE] = getElementSize();

	glGenTextures (1, &(m_UIntProps[ID]));
	glBindTexture (m_EnumProps[DIMENSION], m_UIntProps[ID]);

	glTexImage2D(m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT], m_IntProps[WIDTH], m_IntProps[HEIGHT], 0,
 		m_EnumProps[FORMAT], m_EnumProps[TYPE], data);

	if (data != NULL) {
		m_BoolProps[MIPMAP] = mipmap;
#ifndef NAU_OPTIX
		if (mipmap)
			glGenerateMipmap(GL_TEXTURE_2D);
#else
		m_BoolProps[MIPMAP] = false;
#endif
	}
	else
		m_BoolProps[MIPMAP] = false;
	glBindTexture (m_EnumProps[DIMENSION], 0);
}


//GLTexture::GLTexture(std::string label) : Texture (label)
//{
//	m_BoolProps[MIPMAP] = false;
//}


GLTexture::~GLTexture(void)
{
	glDeleteTextures (1, &( m_UIntProps[ID]));
}


/// As is, it is useless unless there is a match between format and internal format
int 
GLTexture::getElementSize() {

	int nComp = getNumberOfComponents();
	switch (m_EnumProps[TYPE]) {
		case GL_FLOAT: 
			return nComp * sizeof(float);
			break;
		case GL_UNSIGNED_INT:
			return nComp * sizeof(unsigned int);
			break;
		case GL_UNSIGNED_SHORT:
			return nComp * sizeof(unsigned short);
			break;
		case GL_UNSIGNED_BYTE:
			return nComp * sizeof(unsigned char);
			break;
		case GL_INT:
			return nComp * sizeof(int);
			break;
		case GL_BYTE:
			return nComp * sizeof(char);
			break;
		case GL_SHORT:
			return nComp * sizeof(short);
			break;

	}
	return 0;
}


void 
GLTexture::prepare(int aUnit, TextureSampler *ts) {

	RENDERER->addTexture((IRenderer::TextureUnit)aUnit, this);
	IRenderer::TextureUnit tu = (IRenderer::TextureUnit)(IRenderer::TEXTURE_UNIT0 + aUnit);
	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(m_EnumProps[DIMENSION],m_UIntProps[ID]);

	ts->prepare(aUnit, m_EnumProps[DIMENSION]);


}


void 
GLTexture::restore(int aUnit) {

	IRenderer::TextureUnit tu = (IRenderer::TextureUnit)(IRenderer::TEXTURE_UNIT0 + aUnit);

	RENDERER->removeTexture((IRenderer::TextureUnit)aUnit);

	glActiveTexture (GL_TEXTURE0+aUnit);
	glBindTexture(m_EnumProps[DIMENSION],0);

	GLTextureSampler::restore(aUnit, m_EnumProps[DIMENSION]);
}


/// Note: this should be based on internal format, not the format
int
GLTexture::getNumberOfComponents(void) {

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


//void 
//GLTexture::setData(std::string anInternalFormat, std::string aFormat, std::string aType, 
//				   int width, int height, unsigned char * data) 
//{
//
//	m_EnumProps[DIMENSION] = GL_TEXTURE_2D;
//	m_EnumProps[FORMAT] = Attribs.getListValueOp(FORMAT, aFormat);
//	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
//	m_EnumProps[TYPE] = Attribs.getListValueOp(TYPE, aType);
//
//	m_IntProps[WIDTH] = width;
//	m_IntProps[HEIGHT] = height;
//	m_IntProps[DEPTH] = 1;
//	m_IntProps[COMPONENT_COUNT] = getNumberOfComponents();
//	m_IntProps[ELEMENT_SIZE] = getElementSize();
//
//	glGenTextures (1, &(m_UIntProps[ID]));
//	glBindTexture (m_EnumProps[DIMENSION],m_UIntProps[ID]);
//	glTexImage2D (m_EnumProps[DIMENSION], 0, m_EnumProps[INTERNAL_FORMAT], 
//						width, height, 0, m_EnumProps[FORMAT], m_EnumProps[TYPE], data);
//
//
//	glBindTexture (m_EnumProps[DIMENSION], 0);
//
//	m_BoolProps[MIPMAP] = false;
//}




//int 
//GLTexture::getIndex(std::string StringArray[], int IntArray[], std::string aString)
//{
//	int i;
//	for (i = 0; (IntArray[i] != GL_INVALID_ENUM) && (StringArray[i].compare(aString)); i++) ;
////		i++;
//	return IntArray[i];
//}


/*
void 
GLTexture::enableCompareToTexture (void)
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	//glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
}

void 
GLTexture::disableCompareToTexture (void)
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_NONE);
}
*/





//void 
//GLTexture::enableObjectSpaceCoordGen (void)
//{
//	glTexGeni (GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
//	glTexGeni (GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
//}
//
//void 
//GLTexture::generateObjectSpaceCoords (TextureCoord aCoord, float *plane)
//{
//	glTexGenfv (translateCoord (aCoord), GL_OBJECT_PLANE, plane);
//}

