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
	if (sup->apiSupport(IAPISupport::CLEAR_BUFFER)) {

		glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
		glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}


void 
GLBuffer::setData(size_t size, void *data) {

	m_UIntProps[SIZE] = (unsigned int)size;
	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glBufferData(GL_ARRAY_BUFFER, m_UIntProps[SIZE], data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void
GLBuffer::setSubData(size_t offset, size_t size, void *data) {

	glBindBuffer(GL_ARRAY_BUFFER, m_IntProps[ID]);
	glBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
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

	//glMemoryBarrier(GL_ALL_BARRIER_BITS);
	GLenum type = GL_UNIFORM_BUFFER;// SHADER_STORAGE_BUFFER;
	glBindBuffer(type, m_IntProps[ID]);
	//glMemoryBarrier(GL_ALL_BARRIER_BITS);
	void *bufferData;
	bufferData = glMapBufferRange(type, offset, actualSize, GL_MAP_READ_BIT);
	assert(bufferData != NULL);
	memcpy(data, bufferData, actualSize);
	glUnmapBuffer(type);
	//glGetBufferSubData(type, offset, actualSize, data);
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

		IAPISupport *sup = IAPISupport::GetInstance();
		if (sup->apiSupport(IAPISupport::CLEAR_BUFFER)) {
			glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
		}		
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