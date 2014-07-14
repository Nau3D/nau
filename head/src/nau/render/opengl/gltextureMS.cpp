#include <nau/render/opengl/gltextureMS.h>


using namespace nau::render;

bool GLTextureMS::Inited = GLTextureMS::InitGL();

bool
GLTextureMS::InitGL() {

	Attribs.listAdd("DIMENSION", "TEXTURE_2D_MULTISAMPLE", GL_TEXTURE_2D_MULTISAMPLE);
	return true;
}

	
	
GLTextureMS::GLTextureMS (std::string label, std::string anInternalFormat, int width, int height, int samples) :
	GLTexture ()
{
	m_Label = label;
	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = height;
	m_IntProps[DEPTH] = 1;
	m_IntProps[SAMPLES] = samples;

	m_EnumProps[DIMENSION] = GL_TEXTURE_2D_MULTISAMPLE;
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
	m_EnumProps[FORMAT] = GLTexture::GetCompatibleFormat(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
	m_EnumProps[TYPE] = GLTexture::GetCompatibleType(m_EnumProps[DIMENSION], m_EnumProps[INTERNAL_FORMAT]);
	m_IntProps[COMPONENT_COUNT] = getNumberOfComponents();
	m_IntProps[ELEMENT_SIZE] = getElementSize();


	glGenTextures (1, &(m_UIntProps[ID]));
	glBindTexture (m_EnumProps[DIMENSION], m_UIntProps[ID]);

	glTexImage2DMultisample(m_EnumProps[DIMENSION], samples, m_EnumProps[INTERNAL_FORMAT] , m_IntProps[WIDTH], m_IntProps[HEIGHT], true);

	m_BoolProps[MIPMAP] = false;

	glBindTexture (m_EnumProps[DIMENSION], 0);
}



GLTextureMS::~GLTextureMS(void)
{
	glDeleteTextures (1, &(m_UIntProps[ID]));
}

