#include <nau/render/vertexdata.h>
#include <nau/config.h>
#include <assert.h>

#ifdef NAU_OPENGL
#include <nau/render/opengl/glvertexarray.h>
#endif

using namespace nau::render;
using namespace nau::math;

std::vector<VertexData::Attr> VertexData::NoData;

const std::string nau::render::VertexData::Syntax[] = {
	"position", 
	"normal", 
	"color",
	//"secondaryColor",
	//"edge",
	//"fogCoord",
	"texCoord0",
	"texCoord1",
	"texCoord2",
	"texCoord3",
	//"texCoord4",
	//"texCoord5",
	//"texCoord6",
	//"texCoord7",
	"tangent",
	"binormal",
	"triangleID",
	"custom0",
	"custom1"
	"custom2",
	"custom3",
	"custom4",
	"custom5",
	"index"};


unsigned int 
VertexData::getAttribIndex(std::string attribName) {

	unsigned int index;

	for (index = 0; index < VertexData::MaxAttribs; index++) {

		if (VertexData::Syntax[index].compare(attribName) == 0)
			return(index);
	}
	assert("Attrib Index out of range");
	return (MaxAttribs);
}


VertexData* 
VertexData::create (void)
{
#ifdef NAU_OPENGL
	return new GLVertexArray;
#elif NAU_DIRECTX
	return new DXVertexArray;
#endif
}


VertexData::VertexData(void) 
{
	for (int i = 0; i < VertexData::MaxAttribs; i++) {
		m_InternalArrays[i] = 0;
	}
}


VertexData::~VertexData(void)
{
	for (int i = 0; i < VertexData::MaxAttribs; i++) {

		if (0 != m_InternalArrays[i]){
			delete m_InternalArrays[i];
		}
	}
}


int 
VertexData::getNumberOfVertices() 
{
	if (m_InternalArrays[0] == NULL)
		return 0;
	else
		return m_InternalArrays[0]->size();
}


std::vector<VertexData::Attr>& 
VertexData::getDataOf (unsigned int type)
{
	if (0 == m_InternalArrays[type]) {
		return VertexData::NoData;
	}
	return (*m_InternalArrays[type]);
}


void 
VertexData::setDataFor (unsigned index, std::vector<Attr>* dataArray)
{
	assert(index < VertexData::MaxAttribs && *dataArray != VertexData::NoData);

	if (0 != m_InternalArrays[index])
		delete m_InternalArrays[index];

	if (index < VertexData::MaxAttribs && *dataArray != VertexData::NoData) {
		m_InternalArrays[index] = dataArray;
	}
}


int
VertexData::add (VertexData &aVertexData)
{
	size_t offset = 0;

	std::vector<Attr> &vertices = getDataOf (getAttribIndex("position"));

	offset = vertices.size();

	if (offset == 0) {

		for (int i = 0; i < VertexData::MaxAttribs; i++) {
			std::vector<Attr> &newVec = aVertexData.getDataOf(i);
			if (newVec != VertexData::NoData) {
				std::vector<Attr> *aVec = new std::vector<Attr>;
				aVec->insert(aVec->end(),newVec.begin(), newVec.end());
				setDataFor(i,aVec);
			}
		}
	}
	else {

		for (int i = 0; i < VertexData::MaxAttribs; i++) {

			std::vector<Attr> &thisVec = getDataOf(i);
			std::vector<Attr> &newVec = aVertexData.getDataOf(i);

			if (newVec != VertexData::NoData && thisVec != VertexData::NoData) {
			
				thisVec.insert(thisVec.end(),newVec.begin(), newVec.end());
			}
			else if (newVec != VertexData::NoData && thisVec == VertexData::NoData) {
			
				std::vector<Attr> *aVec = new std::vector<Attr>(offset);
				aVec->insert(aVec->end(), newVec.begin(), newVec.end());
				setDataFor(i,aVec);
			}
			else if (newVec == VertexData::NoData && thisVec != VertexData::NoData) {

				int size = aVertexData.getDataOf(getAttribIndex("position")).size();
				thisVec.resize(offset + size);
			}
		}
	}



	return (int)offset;
}


void
VertexData::unitize(float min, float max) {

	std::vector<Attr> &vertices = getDataOf (getAttribIndex("position"));
	unsigned int i;

	if (vertices == VertexData::NoData) 
		return;

	//scale vertices so that min = -1 and max = 1
	for( i = 0; i < vertices.size(); i++) {
	
		vertices[i].x = ((vertices[i].x - min)/(max - min) * 2.0f) - 1.0f;
		vertices[i].y = ((vertices[i].y - min)/(max - min) * 2.0f) - 1.0f;
		vertices[i].z = ((vertices[i].z - min)/(max - min) * 2.0f) - 1.0f;
	}
}


