#include <nau/render/opengl/gltexture2dArray.h>


using namespace nau::render;

bool GLTexture2DArray::Inited = GLTexture2DArray::InitGL();

bool
GLTexture2DArray::InitGL() {

	Attribs.listAdd("DIMENSION", "TEXTURE_2D_ARRAY", GL_TEXTURE_2D_ARRAY);
	return true;
}

	
	
GLTexture2DArray::GLTexture2DArray (std::string label, std::string anInternalFormat, int width, int height, int layers) :
	GLTexture ()
{
	m_Label = label;
	m_IntProps[WIDTH] = width;
	m_IntProps[HEIGHT] = height;
	m_IntProps[DEPTH] = layers;
	m_IntProps[LAYERS] = layers;

	m_EnumProps[DIMENSION] = GL_TEXTURE_2D_ARRAY;
	m_EnumProps[FORMAT] = GL_RGBA;
	m_EnumProps[INTERNAL_FORMAT] = Attribs.getListValueOp(INTERNAL_FORMAT, anInternalFormat);
	m_EnumProps[TYPE] = GL_FLOAT;
	m_IntProps[COMPONENT_COUNT] = getNumberOfComponents();
	m_IntProps[ELEMENT_SIZE] = getElementSize();


	glGenTextures (1, &(m_UIntProps[ID]));
	glBindTexture (m_EnumProps[DIMENSION], m_UIntProps[ID]);

	glTexStorage3D(m_EnumProps[DIMENSION], 1, m_EnumProps[INTERNAL_FORMAT] , m_IntProps[WIDTH], m_IntProps[HEIGHT], layers);

	m_BoolProps[MIPMAP] = false;

	glBindTexture (m_EnumProps[DIMENSION], 0);
}



GLTexture2DArray::~GLTexture2DArray(void)
{
	glDeleteTextures (1, &(m_UIntProps[ID]));
}

