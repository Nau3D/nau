#include "nau/render/opengl/glBuffer.h"

#include "nau/render/iAPISupport.h"

using namespace nau::render;

std::map<GLenum, GLenum> GLBuffer::BufferBound = {
	{ GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING },
	{ GL_ELEMENT_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER_BINDING },
	{ GL_DRAW_INDIRECT_BUFFER, GL_DRAW_INDIRECT_BUFFER_BINDING },
	{ GL_ATOMIC_COUNTER_BUFFER, GL_ATOMIC_COUNTER_BUFFER_BINDING },
	{ GL_TRANSFORM_FEEDBACK_BUFFER, GL_TRANSFORM_FEEDBACK_BUFFER_BINDING },
	{ GL_UNIFORM_BUFFER, GL_UNIFORM_BUFFER_BINDING },
	{ GL_SHADER_STORAGE_BUFFER, GL_SHADER_STORAGE_BUFFER_BINDING },
	{ GL_ATOMIC_COUNTER_BUFFER, GL_ATOMIC_COUNTER_BUFFER_BINDING }
};


bool
GLBuffer::Init() {

	return true;
}


bool GLBuffer::Inited = Init();


GLBuffer::GLBuffer(std::string label): IBuffer(), m_LastBound((int)GL_ARRAY_BUFFER) {

	IAPISupport *sup = IAPISupport::GetInstance();

	m_Label = label;
	m_BufferMapPointer = NULL;
	glGenBuffers(1, (GLuint *)&m_IntProps[ID]);
	
	if (sup->apiSupport(IAPISupport::OBJECT_LABELS)) {
		glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
		glObjectLabel(GL_BUFFER, m_IntProps[ID], (GLsizei)m_Label.size(), m_Label.c_str());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
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
	glBindBuffer((GLenum)target, m_IntProps[ID]);
}


void
GLBuffer::unbind() {

	glBindBuffer((GLenum)m_LastBound, 0);
}


void 
GLBuffer::clear() {

	IAPISupport *sup = IAPISupport::GetInstance();

	if (sup->apiSupport(IAPISupport::DIRECT_ACCESS)) {

		glClearNamedBufferData(m_IntProps[ID], GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
	}
	else if (sup->apiSupport(IAPISupport::CLEAR_BUFFER)) {

		glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
		glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}


void 
GLBuffer::setData(size_t size, void *data) {

	IAPISupport *sup = IAPISupport::GetInstance();

	m_UIntProps[SIZE] = (unsigned int)size;

	if (sup->apiSupport(IAPISupport::DIRECT_ACCESS)) {
		glNamedBufferData(m_IntProps[ID], m_UIntProps[SIZE], data, GL_STATIC_DRAW);
	}
	else {
		glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
		glBufferData(GL_ARRAY_BUFFER, m_UIntProps[SIZE], data, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}


void
GLBuffer::setSubData(size_t offset, size_t size, void *data) {

	IAPISupport *sup = IAPISupport::GetInstance();

	if (sup->apiSupport(IAPISupport::DIRECT_ACCESS)) {
		glNamedBufferSubData(m_IntProps[ID], offset, size, data);
	}
	else {
		glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
		glBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}


void
GLBuffer::setSubDataNoBinding(unsigned int bufferType, size_t offset, size_t size, void *data) {

	glBufferSubData((GLenum)bufferType, offset, size, data);
}


size_t
GLBuffer::getData(size_t offset, size_t size, void *data) {

	size_t actualSize = size;

	if (offset >= m_UIntProps[SIZE])
		return 0;
	
	if (offset + size > m_UIntProps[SIZE])
		actualSize = (size_t)m_UIntProps[SIZE] - offset;

	GLenum type = GL_ARRAY_BUFFER;

	IAPISupport *sup = IAPISupport::GetInstance();

	if (sup->apiSupport(IAPISupport::DIRECT_ACCESS)) {
		glGetNamedBufferSubData(m_IntProps[ID], offset, actualSize, data);
	}
	else {
		glBindBuffer(type, m_IntProps[ID]);
		glGetBufferSubData(type, offset, actualSize, data);
		glBindBuffer(type, 0);
	}
	return actualSize;
}


void
GLBuffer::setPropui(UIntProperty  prop, unsigned int value) {

	IAPISupport *sup = IAPISupport::GetInstance();

	if (prop == SIZE) {
		m_UIntProps[SIZE] = value;

		if (sup->apiSupport(IAPISupport::DIRECT_ACCESS)) {
			glNamedBufferData(m_IntProps[ID], m_UIntProps[SIZE], NULL, GL_STATIC_DRAW);
			glClearNamedBufferData(m_IntProps[ID], GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
		}
		else {
			glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
			glBufferData(GL_ARRAY_BUFFER, m_UIntProps[SIZE], NULL, GL_STATIC_DRAW);

			if (sup->apiSupport(IAPISupport::CLEAR_BUFFER)) {
				glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
			}
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
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

	IAPISupport *sup = IAPISupport::GetInstance();

	int value;

	if (sup->apiSupport(IAPISupport::DIRECT_ACCESS)) {
		glGetNamedBufferParameteriv(m_IntProps[ID], GL_BUFFER_SIZE, &value);
	}
	else {
		glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
		glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &value);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	m_UIntProps[SIZE] = (unsigned int)value;
}