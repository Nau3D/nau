#include "nau/render/opengl/glIndexArray.h"

#include "nau.h"

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>
//#include <GL/gl.h>

#include <assert.h>

using namespace nau::render;
using namespace nau::math;


//GLIndexArray::GLIndexArray(void):
//	IndexData (),
//	m_GLBuffer(0),
//	m_IsCompiled (false) {
//
//}

GLIndexArray::GLIndexArray(std::string & name):
	IndexData(name),
	m_IsCompiled(false) {

}


GLIndexArray::~GLIndexArray(void) {

	if (0 != m_BufferID) {
		glDeleteBuffers (1, &m_BufferID);
	}
}


unsigned int
GLIndexArray::getBufferID() {

	return 	m_BufferID;
}


void
GLIndexArray::setBuffer(unsigned int id) {

	m_BufferID = id;
	if (m_BufferID != 0)
		m_IsCompiled = true;

}


//std::vector<unsigned int>&
//GLIndexArray::getIndexData (void) {
//
//	if (0 == m_InternalIndexArray) { 
//		return IndexData::NoIndexData;
//	}
//	return (*m_InternalIndexArray);
//}


//bool 
//GLIndexArray::compile (VertexData &v) {
//
//	if (m_IsCompiled)
//		return false;
//
//	m_IsCompiled = true;
//
//	if (!v.isCompiled())
//		v.compile();
//
//	glGenVertexArrays(1, &m_VAO);
//	glBindVertexArray(m_VAO);
//
//	v.bind();
////	glBindVertexArray(0);
//
//	if (m_GLBuffer != 0)
//		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_GLBuffer);
//
//	else if (0 != m_InternalIndexArray && m_InternalIndexArray->size() != 0) {
//
//		std::vector<unsigned int>* pArray;
//		if (m_UseAdjacency) {
//			buildAdjacencyList();
//			pArray = &m_AdjIndexArray;
//		}
//		else
//			pArray = m_InternalIndexArray;
//
//		std::string s;
//		
//		IBuffer *b = RESOURCEMANAGER->createBuffer(m_Name);
//		b->setStructure(std::vector<Enums::DataType>{Enums::UINT});
//		m_GLBuffer = b->getPropi(IBuffer::ID);
//		glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_GLBuffer);
//		glBufferData (GL_ELEMENT_ARRAY_BUFFER, pArray->size() * sizeof (unsigned int), &(*pArray)[0], GL_STATIC_DRAW);
//	}
//
//	glBindVertexArray(0);
//	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);	
//	v.unbind();
//
//	return true;
//}


void
GLIndexArray::compile() {

	m_IsCompiled = true;

	if (0 != m_InternalIndexArray && m_InternalIndexArray->size() != 0) {

		std::shared_ptr<std::vector<unsigned int>> pArray;
		if (m_UseAdjacency) {
			buildAdjacencyList();
			pArray = m_AdjIndexArray;
		}
		else
			pArray = m_InternalIndexArray;

		IBuffer *b = NULL;

		if (m_BufferID == 0) {
			b = RESOURCEMANAGER->createBuffer(m_Name);
			b->setStructure(std::vector < Enums::DataType > {Enums::UINT});
			m_BufferID = b->getPropi(IBuffer::ID);
		}
		else {
			b = RESOURCEMANAGER->getBuffer(m_Name);
		}
		b->setData(pArray->size() * sizeof(unsigned int), &(*pArray)[0]);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_GLBuffer);
		//glBufferData(GL_ELEMENT_ARRAY_BUFFER, pArray->size() * sizeof(unsigned int), &(*pArray)[0], GL_STATIC_DRAW);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		//b->setPropui(IBuffer::SIZE, pArray->size() * sizeof(unsigned int));
	}
}

void 
GLIndexArray::resetCompilationFlag() {

	m_IsCompiled = false;

//	glDeleteBuffers(1, &m_GLBuffer);
}


bool
GLIndexArray::isCompiled() {

	return (m_IsCompiled);
}


void 
GLIndexArray::bind (void) {

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_BufferID);
}


void 
GLIndexArray::unbind (void) {

	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);
}


void 
GLIndexArray::useAdjacency(bool b) {

	m_UseAdjacency = b;
}


bool
GLIndexArray::getAdjacency() {

	return (m_UseAdjacency);
}


unsigned int
GLIndexArray::getIndexSize(void) {

	if (m_InternalIndexArray && m_UseAdjacency == false)
		return (unsigned int)m_InternalIndexArray->size();
	else if (m_AdjIndexArray)
		return (unsigned int)m_AdjIndexArray->size();
	else if (m_BufferID)
		return RESOURCEMANAGER->getBufferByID(m_BufferID)->getPropui(IBuffer::SIZE) / sizeof(unsigned int);
	else
		return 0;
}