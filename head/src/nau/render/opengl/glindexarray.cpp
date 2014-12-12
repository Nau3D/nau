#include <nau/render/opengl/glindexarray.h>

#include <nau.h>

#include <assert.h>

using namespace nau::render;
using namespace nau::math;


GLIndexArray::GLIndexArray(void):
	IndexData (),
	m_VAO(0),
	m_IsCompiled (false)
{
}


GLIndexArray::~GLIndexArray(void)
{
	if (0 != m_InternalIndexArray) {
		delete m_InternalIndexArray;
		m_InternalIndexArray = 0;
	}

	if (0 != m_GLBuffer) {
		glDeleteBuffers (1, &m_GLBuffer);
	}

	glDeleteVertexArrays(1,&m_VAO);
}


std::vector<unsigned int>&
GLIndexArray::getIndexData (void)
{
	if (0 == m_InternalIndexArray) { // || true == m_IsCompiled) {
		return IndexData::NoIndexData;
	}
	return (*m_InternalIndexArray);
}

bool 
GLIndexArray::compile (VertexData &v) 
{
	if (m_IsCompiled)
		return false;

	if (!v.isCompiled())
		v.compile();

	m_IsCompiled = true;

	glGenVertexArrays(1, &m_VAO);
	glBindVertexArray(m_VAO);

	v.bind();
//	glBindVertexArray(0);

	if (0 != m_InternalIndexArray && m_InternalIndexArray->size() != 0) {

		std::vector<unsigned int>* pArray;
		if (m_UseAdjacency) {
			buildAdjacencyList();
			pArray = &m_AdjIndexArray;
		}
		else
			pArray = m_InternalIndexArray;

		std::string s;
		
		IBuffer *b = RESOURCEMANAGER->createBuffer(m_Name);
		b->setStructure(std::vector<Enums::DataType>{Enums::UINT});
		m_GLBuffer = b->getPropi(IBuffer::ID);
		glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_GLBuffer);
		glBufferData (GL_ELEMENT_ARRAY_BUFFER, pArray->size() * sizeof (unsigned int), &(*pArray)[0], GL_STATIC_DRAW);
	}

	glBindVertexArray(0);
	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);	
	v.unbind();

	return true;
}


void 
GLIndexArray::resetCompilationFlag() {

	m_IsCompiled = false;

	glDeleteBuffers(1, &m_GLBuffer);
	glDeleteVertexArrays(1, &m_VAO);

}

bool
GLIndexArray::isCompiled()
{
	return (m_IsCompiled);
}

void 
GLIndexArray::bind (void)
{
	if (true == m_IsCompiled) {
		if (m_VAO)
			glBindVertexArray(m_VAO);
		//else if (0 != m_GLBuffer) {
		//	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_GLBuffer);
		//}
	} 
}





void 
GLIndexArray::unbind (void)
{
	if (m_VAO)
		glBindVertexArray(0);
	else
		glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);
}


//std::vector<unsigned int>& 
//GLIndexArray::_getReallyIndexData (void)
//{
//	return (*m_InternalIndexArray);
//}


unsigned int 
GLIndexArray::getBufferID() {

	return m_GLBuffer;
}


void 
GLIndexArray::useAdjacency(bool b) {

	m_UseAdjacency = b;
}


bool
GLIndexArray::getAdjacency() {

	return (m_UseAdjacency);
}