#include "nau/render/opengl/glUniformBlock.h"

#include "nau.h"

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>


using namespace nau::render;

GLUniformBlock::GLUniformBlock(): 
	m_Size(0), m_LocalData(NULL), m_BlockChanged(true) { 

}


GLUniformBlock::GLUniformBlock(std::string &name, unsigned int size) {

	m_Buffer = RESOURCEMANAGER->createBuffer(name);
	m_Size = size;
	m_LocalData = (void *)malloc(size);
	m_BindingIndex = 0;
	m_BlockChanged = true;
}


GLUniformBlock::~GLUniformBlock() {

	if (m_LocalData)
		free(m_LocalData);
}


void 
GLUniformBlock::init(std::string &name, unsigned int size) {

	m_Buffer = RESOURCEMANAGER->createBuffer(name);
	m_Size = size;
	m_LocalData = (void *)malloc(size);
}


void
GLUniformBlock::setBindingIndex(unsigned int i) {

	m_BindingIndex = i;
}


void 
GLUniformBlock::addUniform(std::string &name, Enums::DataType type, unsigned int offset, 
							unsigned int size, unsigned int arrayStride) {

	blockUniform b;

	// Uniform already there?
	if (m_Uniforms.count(name))
		return;

	b.type = type;
	b.offset = offset;
	
	if (size == 0) {
		b.size = Enums::getSize(type);
	}
	else {
		b.size = size;
	}

	b.arrayStride = arrayStride;

	m_Uniforms[name] = b;
}


void 
GLUniformBlock::setUniform(std::string &name, void *value) {

	assert(m_Uniforms.count(name));

	blockUniform b = m_Uniforms[name];
	if (memcmp((char *)m_LocalData + b.offset, value, b.size)) {
		memcpy((char *)m_LocalData + b.offset, value, b.size);
		m_BlockChanged = true;
	}
}


void 
GLUniformBlock::setBlock(void *value) {

	memcpy((char *)m_LocalData , value, m_Size);
	m_BlockChanged = true;
}


void 
GLUniformBlock::sendBlockData() {

	if (m_BlockChanged) {
		m_Buffer->setSubData(m_Size, 0, m_LocalData);
		m_BlockChanged = false;
	}
}


void 
GLUniformBlock::useBlock() {

	m_Buffer->bind((unsigned int)GL_UNIFORM_BUFFER);
	if (m_BlockChanged) {
		m_Buffer->setSubDataNoBinding((unsigned int)GL_UNIFORM_BUFFER, 0, m_Size, m_LocalData);
		m_BlockChanged = false;
	}
}


unsigned int
GLUniformBlock::getSize() {

	return m_Size;
}


bool 
GLUniformBlock::hasUniform(std::string &name) {

	if (m_Uniforms.count(name))
		return true;
	else
		return false;
}


Enums::DataType 
GLUniformBlock::getUniformType(std::string name) {

	assert(m_Uniforms.count(name));

	return m_Uniforms[name].type;
}


IBuffer *
GLUniformBlock::getBuffer() {

	return m_Buffer;
}