#include "nau/render/opengl/glVertexArray.h"

#include "nau.h"
#include "nau/render/opengl/glBuffer.h"

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>

#include <assert.h>

using namespace nau::render;
using namespace nau::math;

//  STATIC METHODS




//	CONST e DEST
GLVertexArray::GLVertexArray(const std::string &name):
	VertexData (name),
	m_IsCompiled (false) {

	for (int i = 0; i < VertexData::MaxAttribs; i++){
		m_GLBuffers[i] = 0;
	}
	for (int i = 0; i < VertexData::MaxAttribs; i++) {
		m_AttributesLocations[i] = VertexData::NOLOC;
	}
}


GLVertexArray::~GLVertexArray(void) {

	for (int i = 0; i < VertexData::MaxAttribs; i++) {
		if (0 != m_GLBuffers[i]) {
			glDeleteBuffers (1, &m_GLBuffers[i]);
		}
	}
}


unsigned int 
GLVertexArray::getBufferID(unsigned int vertexAttrib) {

	if (vertexAttrib > VertexData::MaxAttribs)
		return 0;
	else {
		if (m_GLBuffers[vertexAttrib] == 0)
			compile();
		return m_GLBuffers[vertexAttrib];
	}
}


void 
GLVertexArray::setBuffer(unsigned int type, int bufferID) {

	if (m_GLBuffers[type] == 0)
		m_GLBuffers[type] = bufferID;
}


unsigned int
GLVertexArray::getNumberOfVertices()
{
	if (m_InternalArrays[0] != NULL)
		return (unsigned int)m_InternalArrays[0]->size();
	else if (m_GLBuffers[0] != 0)
		return RESOURCEMANAGER->getBufferByID(m_GLBuffers[0])->getPropui(IBuffer::SIZE)/16;
	else
		return 0;
}



void 
GLVertexArray::setAttributeDataFor (unsigned int type, std::vector<VertexData::Attr>* dataArray, int location) {

	if (*dataArray != VertexData::NoData) {
		m_InternalArrays[type] = dataArray;
		if (NOLOC == m_AttributesLocations[type]) {
			m_AttributesLocations[type] = location;
		}
	}
}


void 
GLVertexArray::setAttributeLocationFor (unsigned int type, int location) {

	m_AttributesLocations[type] = location;
}


// must be called prior to COMPILING!
// attributes must be bound prior to linking program
// tangent = 0 and triangleIDs = 1
void 
GLVertexArray::prepareTriangleIDs(unsigned int sceneObjID, 
											 unsigned int primitiveOffset, 
											 std::vector<unsigned int> *index) 
{
	unsigned int TriID_index = VertexData::GetAttribIndex(std::string("triangleID"));

	if (sceneObjID != 0 && 0 == m_InternalArrays[TriID_index] 
						 && m_InternalArrays[0]) {

		unsigned int size = (unsigned int)m_InternalArrays[0]->size();
		m_InternalArrays[TriID_index] = new std::vector<VertexData::Attr>(size);

		unsigned int begin;
		if (primitiveOffset != 1) // no strips or fans
			begin = 0;
		else
			begin = 2;
		unsigned int i,k,j;
		for (i = begin, j = 0; i < size-primitiveOffset+1; i+=primitiveOffset, j++) {
			for (k = 0; k < primitiveOffset; k++) 
				m_InternalArrays[TriID_index]->
								at(i + k  ).set((float)sceneObjID,(float)j,0.0,0.0);

		}
		for (unsigned int i = 0; i < size; i++)
			m_InternalArrays[TriID_index]->at(i  ).x = (float)sceneObjID;

		m_AttributesLocations[TriID_index] = 1;
	}
}


void
GLVertexArray::appendVertex(unsigned int index) {

	for (int i = 0; i < VertexData::MaxAttribs; i++) {

		if (m_InternalArrays[i] != 0) {

			VertexData::Attr v;
			v.copy(m_InternalArrays[i]->at(index));
			m_InternalArrays[i]->push_back(v);
		}
	}
}


bool
GLVertexArray::isCompiled() {

		return m_IsCompiled;
}


void
GLVertexArray::resetCompilationFlag() {

	m_IsCompiled = false;

	//for (int i = 0; i < VertexData::MaxAttribs; i++){
	//	if (m_GLBuffers[i]) {
	//		glDeleteBuffers(1, &m_GLBuffers[i]);
	//		m_GLBuffers[i] = 0;
	//	}
	//}
}


bool 
GLVertexArray::compile (void) { 

	if (m_IsCompiled)
		return false;

	m_IsCompiled = true;

	for (int i = 0; i < VertexData::MaxAttribs; i++){

		if (0 != m_InternalArrays[i]){
			std::vector<VertexData::Attr>* pArray = m_InternalArrays[i];

			IBuffer *b = NULL;

			std::string s = m_Name + ":" + VertexData::Syntax[i];
			if (m_GLBuffers[i] == 0) {
				b = RESOURCEMANAGER->createBuffer(s);
				b->setStructure(std::vector < Enums::DataType > {Enums::FLOAT, Enums::FLOAT, Enums::FLOAT, Enums::FLOAT});
				m_GLBuffers[i] = b->getPropi(IBuffer::ID);
			}
			else {
				b = RESOURCEMANAGER->getBuffer(s);
			}
			b->setData(pArray->size() * 4 * sizeof(float), (float*)&(*pArray)[0]);
			//glBindBuffer(GL_ARRAY_BUFFER, m_GLBuffers[i]);
			//glBufferData(GL_ARRAY_BUFFER, pArray->size() * 4 * sizeof(float), (float*)&(*pArray)[0], GL_STATIC_DRAW);
			//glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
	}
	return true;
}


void 
GLVertexArray::bind (void) {

	unsigned int i;

	if (true == m_IsCompiled) {
		for (i = 0; i < VertexData::MaxAttribs; ++i) {
			if (0 != m_GLBuffers[i]){

				int loc = RENDERER->getAttribLocation(std::string(VertexData::Syntax[i]));
				if (loc != -1) {
					glBindBuffer (GL_ARRAY_BUFFER, m_GLBuffers[i]);
					glEnableVertexAttribArray (loc/*m_AttributesLocations[i]*/);
					glVertexAttribPointer (loc/*m_AttributesLocations[i]*/, 4, GL_FLOAT, GL_FALSE, 0, 0);
				}
			}
		}
	} 
}


void 
GLVertexArray::unbind (void) {

	unsigned int i;

	if (true == m_IsCompiled) {

		for (i = 0; i < VertexData::MaxAttribs; i++) {
			if (0 != m_GLBuffers[i]) {
				glDisableVertexAttribArray (m_AttributesLocations[i]);
			}
		}
	} 
	glBindBuffer (GL_ARRAY_BUFFER, 0);
	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

	glActiveTexture (GL_TEXTURE0);
}
