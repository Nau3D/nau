#include <GL/glew.h>
#include <nau/render/opengl/glbuffer.h>


using namespace nau::render;

bool
GLBuffer::Init() {

	return true;
}


bool GLBuffer::Inited = Init();


GLBuffer::GLBuffer(std::string label): m_LastBound(GL_ARRAY_BUFFER) {

	initArrays(Attribs);
	m_Label = label;
	glGenBuffers(1, (GLuint *)&m_IntProps[ID]);
#if NAU_OPENGL_VERSION >= 430
	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glObjectLabel(GL_BUFFER, m_IntProps[ID], m_Label.size(), m_Label.c_str());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
}


GLBuffer::~GLBuffer() {

	glDeleteBuffers(1, (GLuint*)&m_IntProps[ID]);
}


IBuffer *
GLBuffer::clone() {

	GLBuffer *b = new GLBuffer();
	b->m_Label = this->m_Label;
	AttributeValues::copy(b);
	return (IBuffer *)b;
}


void
GLBuffer::bind(unsigned int target) {

	m_LastBound = target;
	glBindBuffer(target, m_IntProps[ID]);
}


void
GLBuffer::unbind() {

	glBindBuffer(m_LastBound, 0);
}


#if NAU_OPENGL_VERSION >= 430
void 
GLBuffer::clear() {

	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
#endif


void 
GLBuffer::setData(unsigned int size, void *data) {

	m_UIntProps[SIZE] = size;
	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glBufferData(GL_ARRAY_BUFFER, m_UIntProps[SIZE], data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void
GLBuffer::setSubData(unsigned int offset, unsigned int size, void *data) {

	m_UIntProps[SIZE] = size;
	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glBufferSubData(GL_ARRAY_BUFFER, offset, m_UIntProps[SIZE], data);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


int
GLBuffer::getData(unsigned int offset, unsigned int size, void *data) {

	int actualSize = size;

	if (offset >= m_UIntProps[SIZE])
		return 0;
	
	if (offset + size > m_UIntProps[SIZE])
		actualSize = m_UIntProps[SIZE] - offset;

	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glGetBufferSubData(GL_ARRAY_BUFFER, offset, actualSize, data);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return actualSize;
}


void
GLBuffer::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {

		case Enums::FLOAT:
			m_FloatProps[prop] = *(float *)value;
			break;
		case Enums::VEC4:
			m_Float4Props[prop].set((vec4 *)value);
			break;
		case Enums::INT:
			m_IntProps[prop] = *(int *)value;
			break;
		case Enums::UINT:
			m_UIntProps[prop] = *(unsigned int *)value;
			if (prop == SIZE) {
				glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
				//glBufferStorage(GL_ARRAY_BUFFER, m_UIntProps[SIZE], NULL, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT);
				glBufferData(GL_ARRAY_BUFFER, m_UIntProps[SIZE], NULL, GL_STATIC_DRAW);
#if NAU_OPENGL_VERSION >= 430
				glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
#endif
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}

			break;
		case Enums::ENUM:
			m_EnumProps[prop] = *(int *)value;
			break;
		case Enums::BOOL:
			m_BoolProps[prop] = *(bool *)value;
			break;
	}
}


void 
GLBuffer::refreshBufferParameters() {

	int value;
	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &value);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	m_UIntProps[SIZE] = (unsigned int)value;
}