#include "nau/render/opengl/glbuffer.h"

#include <GL/glew.h>

using namespace nau::render;

bool
GLBuffer::Init() {

	return true;
}


bool GLBuffer::Inited = Init();


GLBuffer::GLBuffer(std::string label): IBuffer(), m_LastBound(GL_ARRAY_BUFFER) {

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
	b->copy(this);
	//AttributeValues::copy(b);
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


void 
GLBuffer::clear() {

#if NAU_OPENGL_VERSION >= 430
	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
}


void 
GLBuffer::setData(unsigned int size, void *data) {

	m_UIntProps[SIZE] = size;
	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glBufferData(GL_ARRAY_BUFFER, m_UIntProps[SIZE], data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void
GLBuffer::setSubData(unsigned int offset, unsigned int size, void *data) {

	//m_UIntProps[SIZE] = size;
	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


int
GLBuffer::getData(unsigned int offset, unsigned int size, void *data) {

	int actualSize = size;

	if (offset >= m_UIntProps[SIZE])
		return 0;
	
	if (offset + size > m_UIntProps[SIZE])
		actualSize = m_UIntProps[SIZE] - offset;

		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
	int type = GL_ARRAY_BUFFER;// SHADER_STORAGE_BUFFER;
	glBindBuffer(type, m_IntProps[ID]);
	//void *bufferData;
	//bufferData = glMapBufferRange(type, offset, actualSize, GL_MAP_READ_BIT);
	//memcpy(data, bufferData, actualSize);
	//glUnmapBuffer(type);
	glGetBufferSubData(type, offset, actualSize, data);
	//glMemoryBarrier(GL_ALL_BARRIER_BITS);
	glBindBuffer(type, 0);

	return actualSize;
}


void
GLBuffer::setPropui(UIntProperty  prop, unsigned int value) {


	if (prop == SIZE) {
		m_UIntProps[SIZE] = value;
		glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
		//glBufferStorage(GL_ARRAY_BUFFER, m_UIntProps[SIZE], NULL, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT);
		glBufferData(GL_ARRAY_BUFFER, m_UIntProps[SIZE], NULL, GL_STATIC_DRAW);
#if NAU_OPENGL_VERSION >= 430
		glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
#endif
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	else
		AttributeValues::setPropui(prop, value);
}


void 
GLBuffer::setPropui3(UInt3Property prop, uivec3 &v) {

	if (prop == DIM) {

		int size = v.x * v.y * v.z;
		int s = 0;
		for (auto t : m_Structure) {
			s += Enums::getSize(t);
		}
		setPropui(SIZE, size * s);
		setPropui(STRUCT_SIZE, s);
	}
	else
		AttributeValues::setPropui3(prop, v);
}

void 
GLBuffer::refreshBufferParameters() {

	int value;
	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &value);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	m_UIntProps[SIZE] = (unsigned int)value;
}