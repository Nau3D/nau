#include <nau/render/opengl/glvertexarray.h>
#include <nau.h>
#include <assert.h>

using namespace nau::render;
using namespace nau::math;

//  STATIC METHODS

//unsigned int GLVertexArray::m_OpenGLOwnAttribs = 0;// VertexData::getAttribIndex("texCoord7") + 1;



//	CONST e DEST
GLVertexArray::GLVertexArray(void):
	VertexData (),
	m_IsCompiled (false)
{
	for (int i = 0; i < VertexData::MaxAttribs; i++){
		m_GLBuffers[i] = 0;
	}
	for (int i = 0; i < VertexData::MaxAttribs; i++) {
		m_AttributesLocations[i] = VertexData::NOLOC;
	}

//	m_OpenGLOwnAttribs = 0;



}


GLVertexArray::~GLVertexArray(void)
{
	//for (int i = 0; i < VertexData::MaxAttribs; i++){
	//	if (0 != m_InternalArrays[i]){
	//		delete m_InternalArrays[i];
	//		m_InternalArrays[i] = 0;
	//	}
	//}

	for (int i = 0; i < VertexData::MaxAttribs; i++) {
		if (0 != m_GLBuffers[i]) {
			glDeleteBuffers (1, &m_GLBuffers[i]);
		}
	}
}



unsigned int 
GLVertexArray::getBufferID(unsigned int vertexAttrib)
{
	if (vertexAttrib > VertexData::MaxAttribs)
		return 0;
	else
		return m_GLBuffers[vertexAttrib];
}




void 
GLVertexArray::setAttributeDataFor (unsigned int type, std::vector<VertexData::Attr>* dataArray, int location)
{	
	if (*dataArray != VertexData::NoData) {
		m_InternalArrays[type] = dataArray;
		if (NOLOC == m_AttributesLocations[type]) {
			m_AttributesLocations[type] = location;
		}
	}
}

void 
GLVertexArray::setAttributeLocationFor (unsigned int type, int location)
{
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
	unsigned int TriID_index = VertexData::getAttribIndex("triangleID");

	if (sceneObjID != 0 && 0 == m_InternalArrays[TriID_index] 
						 && m_InternalArrays[0]) {


		unsigned int size = m_InternalArrays[0]->size();
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
GLVertexArray::appendVertex(unsigned int index) 
{
	for (int i = 0; i < VertexData::MaxAttribs; i++) {

		if (m_InternalArrays[i] != 0) {

			vec4 v;
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

	for (int i = 0; i < VertexData::MaxAttribs; i++){
		if (m_GLBuffers[i]) {
			glDeleteBuffers(1, &m_GLBuffers[i]);
			m_GLBuffers[i] = 0;
		}
	}
}


bool 
GLVertexArray::compile (void) /***MARK***/ //STATIC DRAW ONLY
{
	if (m_IsCompiled)
		return false;

	m_IsCompiled = true;

	for (int i = 0; i < VertexData::MaxAttribs; i++){

		if (0 != m_InternalArrays[i]){
			std::vector<VertexData::Attr>* pArray = m_InternalArrays[i];

			glGenBuffers (1, &m_GLBuffers[i]);
			glBindBuffer (GL_ARRAY_BUFFER, m_GLBuffers[i]);
			glBufferData (GL_ARRAY_BUFFER, pArray->size() * 4 * sizeof (float), (float*) &(*pArray)[0], GL_STATIC_DRAW);
			glBindBuffer (GL_ARRAY_BUFFER, 0);

				// don't delete the positions array
				//if (0 != i) {
					//delete pArray;
					//m_InternalArrays[i] = 0;
				//}
			//}
		}
	}
	
	//if (0 != m_InternalIndexArray && m_IndexSize != 0) {
	//	std::vector<unsigned int>* pArray = m_InternalIndexArray;

	//	glGenBuffers (1, &m_GLBuffers[VertexData::MaxAttribs]);
	//	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_GLBuffers[VertexData::MaxAttribs]);
	//	glBufferData (GL_ELEMENT_ARRAY_BUFFER, pArray->size() * sizeof (unsigned int), &(*pArray)[0], GL_STATIC_DRAW);
	//	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);	

	//	//delete pArray;
	//	//m_InternalIndexArray = 0; /***MARK***/ //MEMORY LEAK ALERT!!!
	//}

	return true;
}


void 
GLVertexArray::bind (void)
{
	unsigned int i;

	if (true == m_IsCompiled) {
		//for (i = 0; i < m_OpenGLOwnAttribs; ++i) {

		//	if (0 != m_GLBuffers[i] ) {

		//		glBindBuffer (GL_ARRAY_BUFFER, m_GLBuffers[i]);
		//		setGLArray (i, 0);
		//		glEnableClientState (translate (i));
		//	}
		//}

		for (i = 0; i < VertexData::MaxAttribs; ++i) {
			if (0 != m_GLBuffers[i]){
				int loc = RENDERER->getAttribLocation(VertexData::Syntax[i]);
				if (loc != -1) {
					glBindBuffer (GL_ARRAY_BUFFER, m_GLBuffers[i]);
					glEnableVertexAttribArray (loc/*m_AttributesLocations[i]*/);
					glVertexAttribPointer (loc/*m_AttributesLocations[i]*/, 4, GL_FLOAT, 0, 0, 0);
				}
			}
		}
	} 
//	else {
//		for (i = 0; i < m_OpenGLOwnAttribs; ++i){
//			if (0 != m_InternalArrays[i]){
//				glEnableClientState (translate (i));
//				//std::vector<vec3>& vec = (*m_InternalArrays[i]);
//				setGLArray (i, (float*) (&(*m_InternalArrays[i])[0]));
//			}
//		}
//		for (i = m_OpenGLOwnAttribs; i < VertexData::MaxAttribs; ++i) {
//			if (0 != m_InternalArrays[i]){
////				if (VertexData::NOLOC != m_AttributesLocations[i]) {
//				int loc = RENDERER->getAttribLocation(VertexData::Syntax[i]);
//				if (loc != -1) {
////					glEnableVertexAttribArray (m_AttributesLocations[i]);
//					glEnableVertexAttribArray(loc);
////					glVertexAttribPointer (m_AttributesLocations[i], 4, GL_FLOAT, 0, 0, (float*) (&(*m_InternalArrays[i])[0]));
//					glVertexAttribPointer (loc, 4, GL_FLOAT, 0, 0, (float*) (&(*m_InternalArrays[i])[0]));
//				}
//			}
//		}
//
//	}
	/* Custom attributes */

	/***MARK***/ // Only vec3 

}

void 
GLVertexArray::unbind (void)
{
	unsigned int i;

	if (true == m_IsCompiled) {

		//for (i = 0; i < m_OpenGLOwnAttribs; i++) {
		//	if (0 != m_GLBuffers[i]) {
		//		glDisableClientState (translate (i));
		//	}
		//}

		for (i = 0; i < VertexData::MaxAttribs; i++) {
			if (0 != m_GLBuffers[i]) {
				glDisableVertexAttribArray (m_AttributesLocations[i]);
			}
		}
	} 
	//else {
	//	for (i = 0; i < m_OpenGLOwnAttribs; i++) {
	//		if (0 != m_InternalArrays[i]) {
	//			glDisableClientState (translate (i));
	//		}
	//	}

	//	for (i = m_OpenGLOwnAttribs; i < VertexData::MaxAttribs; i++) {
	//		if (0 != m_InternalArrays[i]) {
	//			glDisableVertexAttribArray (m_AttributesLocations[i]);
	//		}
	//	}	
	//}
	glBindBuffer (GL_ARRAY_BUFFER, 0);
	glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

	glActiveTexture (GL_TEXTURE0);
//	glClientActiveTexture (GL_TEXTURE0);
}

//void
//GLVertexArray::setGLArray (unsigned int type, float* pointer)
//{
//	std::string s = VertexData::Syntax[type];
//
//	if (s.compare("position") == 0)
//			glVertexPointer (4, GL_FLOAT, 0, pointer);
//	else if (s.compare("normal") == 0) 
//			glNormalPointer (GL_FLOAT, sizeof(float)*4, pointer);
//	else if (s.compare("color") == 0) 
//			glColorPointer (4, GL_FLOAT, 0, pointer);
//	else if (s.compare("secondaryColor") == 0 )
//			glSecondaryColorPointer (3, GL_FLOAT, sizeof(float)*4, pointer);
//	else if (s.compare("edge") == 0 )
//			glEdgeFlagPointer (sizeof(float)*4, pointer);
//	else if (s.compare("fogCoord") == 0) 
//			glFogCoordPointer (GL_FLOAT, sizeof(float)*4, pointer);
//	else if (s.compare("texCoord0") == 0) {
//			glActiveTexture (GL_TEXTURE0);
//			glClientActiveTexture (GL_TEXTURE0);
//			glTexCoordPointer (4, GL_FLOAT, 0, pointer);
//		}
//	else if (s.compare("texCoord1") == 0) {
//			glActiveTexture (GL_TEXTURE1);
//			glClientActiveTexture (GL_TEXTURE1);
//			glTexCoordPointer (4, GL_FLOAT, 0, pointer);
//		}
//	else if (s.compare("texCoord2") == 0) {
//			glActiveTexture (GL_TEXTURE2);
//			glClientActiveTexture (GL_TEXTURE2);
//			glTexCoordPointer (4, GL_FLOAT, 0, pointer);
//		}
//	else if (s.compare("texCoord3") == 0) {
//			glActiveTexture (GL_TEXTURE3);
//			glClientActiveTexture (GL_TEXTURE3);
//			glTexCoordPointer (4, GL_FLOAT, 0, pointer);
//		}
//	else if (s.compare("texCoord4") == 0) {
//			glActiveTexture (GL_TEXTURE4);
//			glClientActiveTexture (GL_TEXTURE4);
//			glTexCoordPointer (4, GL_FLOAT, 0, pointer);
//		}
//	else if (s.compare("texCoord5") == 0) {
//			glActiveTexture (GL_TEXTURE5);
//			glClientActiveTexture (GL_TEXTURE5);
//			glTexCoordPointer (4, GL_FLOAT, 0, pointer);
//		}
//	else if (s.compare("texCoord6") == 0) {
//			glActiveTexture (GL_TEXTURE6);
//			glClientActiveTexture (GL_TEXTURE6);
//			glTexCoordPointer (4, GL_FLOAT, 0, pointer);
//		}
//	else if (s.compare("texCoord7") == 0) {
//			glActiveTexture (GL_TEXTURE7);
//			glClientActiveTexture (GL_TEXTURE7);
//			glTexCoordPointer (4, GL_FLOAT, 0, pointer);
//	
//		}
//	else {
//		assert("Error Translating Vertex Type");
//	}
//}

//GLenum
//GLVertexArray::translate (unsigned int type)
//{
//	std::string s = VertexData::Syntax[type];
//
//	if (s.compare("position") == 0)
//	{
//		return GL_VERTEX_ARRAY;
//	}
//	else if (s.compare("normal") == 0)
//	{
//		return GL_NORMAL_ARRAY;
//	}
//	else if (s.compare("color") == 0)
//	{
//		return GL_COLOR_ARRAY;
//	}
//	else if (s.compare("secondaryColor") == 0)
//	{
//		return GL_SECONDARY_COLOR_ARRAY;
//	}
//	else if (s.compare("index") == 0)
//	{
//		return GL_INDEX_ARRAY;
//	}
//	else if (s.compare("edge") == 0)
//	{
//		return GL_EDGE_FLAG_ARRAY;
//	}
//	else if (s.compare("fogCoord") == 0)
//	{
//		return GL_FOG_COORD_ARRAY;
//	}
//	else if (s.compare(0, 8, "texCoord") == 0)
//	{
//		return GL_TEXTURE_COORD_ARRAY;
//	}
//	else {
//		assert("Error Translating Vertex Type");
//		return 0;
//	}
//
//}


//void 
//GLVertexArray::setCore(bool flag)
//{
//	if (flag)
//		m_OpenGLOwnAttribs = 0;
//	else
//		m_OpenGLOwnAttribs = VertexData::getAttribIndex("texCoord7")+1;
//}

// INSTANCE METHODS


//std::vector<VertexData::Attr>& 
//GLVertexArray::getAttributeDataOf (VertexDataType type)
//{
//	if (type < CUSTOM_ATTRIBUTE_ARRAY0 || type > CUSTOM_ATTRIBUTE_ARRAY7) {
//		return VertexData::NoData;
//	}
//	/***MARK***/ //I will assume that the array is not compiled. Otherwise it is necessary to lock the array on the graphic card
//	if (0 == m_InternalArrays[type]) {
//		return VertexData::NoData;
//	}
//	return (*m_InternalArrays[type]);
//}


//std::vector<VertexData::Attr>& 
//GLVertexArray::getAttributeDataOf (unsigned int type)
//{
//	if (type < CUSTOM_ATTRIBUTE_ARRAY0 || type > CUSTOM_ATTRIBUTE_ARRAY7) {
//		return VertexData::NoData;
//	}
//	/***MARK***/ //I will assume that the array is not compiled. Otherwise it is necessary to lock the array on the graphic card
//	if (0 == m_InternalArrays[type]) {
//		return VertexData::NoData;
//	}
//	return (*m_InternalArrays[type]);
//}


//std::vector<unsigned int>&
//GLVertexArray::getIndexData (void)
//{
//	if (0 == m_InternalIndexArray || true == m_IsCompiled) {
//		return VertexData::NoIndexData;
//	}
//	return (*m_InternalIndexArray);
//}

//void 
//GLVertexArray::setAttributeDataFor (VertexDataType type, std::vector<VertexData::Attr>* dataArray, int location)
//{
//	if (type < CUSTOM_ATTRIBUTE_ARRAY0 || type > CUSTOM_ATTRIBUTE_ARRAY7) {
//		return;
//	}
//	
//	if (*dataArray != VertexData::NoData) {
//		m_InternalArrays[type] = dataArray;
//		if (NOLOC == m_AttributesLocations[type - CUSTOM_ATTRIBUTE_ARRAY0]) {
//			m_AttributesLocations[type - CUSTOM_ATTRIBUTE_ARRAY0] = location;
//		}
//	}
//}
//void 
//GLVertexArray::setAttributeLocationFor (VertexDataType type, int location)
//{
//	if (type < CUSTOM_ATTRIBUTE_ARRAY0 || type > CUSTOM_ATTRIBUTE_ARRAY7) {
//		return;
//	}
//	m_AttributesLocations[type - CUSTOM_ATTRIBUTE_ARRAY0] = location;
//}


//void
//GLVertexArray::prepareTangents() 
//{
//	unsigned index = VertexData::getAttribIndex("tangent");
//	if (0 != m_InternalArrays[index])
//		m_AttributesLocations[index] = 0;
//	
//}


//void 
//GLVertexArray::bind (unsigned int buffers)
//{
//	if (true == m_IsCompiled) {
//		int i;
//		unsigned int markedBuffer;
//		for (markedBuffer = DRAW_TEXTURE_COORDS, i = TEXTURE_COORD_ARRAY0; 
//				i >= VERTEX_ARRAY; 
//				markedBuffer >>= 1, i--){
//			if (0 != m_GLBuffers[i]){ /***MARK***/ //URGENT CLEAN UP!!!
//				if (buffers & markedBuffer) {
//					if (INDEX_ARRAY != i){
//						if (TEXTURE_COORD_ARRAY0 == i) {
//							for (int i = TEXTURE_COORD_ARRAY0; i <= TEXTURE_COORD_ARRAY7; i++) {
//								if (0 != m_GLBuffers[i]) {
//									glBindBuffer (GL_ARRAY_BUFFER, m_GLBuffers[i]);
//									setGLArray ((VertexDataType)i, 0);
//								}
//							}
//						} else {
//							glBindBuffer (GL_ARRAY_BUFFER, m_GLBuffers[i]);
//							setGLArray ((VertexDataType)i, 0);
//						}
//					} else {
//						glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_GLBuffers[i]);
//					}
//					glEnableClientState (translate ((VertexDataType)i));
//				}
//			}
//		}
//	} else {
//		for (int i = VERTEX_ARRAY; i <= TEXTURE_COORD_ARRAY7; i++){
//			if (0 != m_InternalArrays[i]){
//				glEnableClientState (translate ((VertexDataType)i));
//				setGLArray ((VertexDataType)i, (float*) (&(*m_InternalArrays[i])[0]));
//			}
//		}
//	}
//}



//std::vector<unsigned int>& 
//GLVertexArray::_getReallyIndexData (void)
//{
//	return (*m_InternalIndexArray);
//}
