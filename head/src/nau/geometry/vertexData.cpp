#include "nau/geometry/vertexData.h"

#include "nau/config.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glVertexArray.h"
#endif

#include <assert.h>

using namespace nau::geometry;
using namespace nau::render;
using namespace nau::math;

const std::string VertexData::Syntax[] = {
	"position", 
	"normal", 
	"color",
	"texCoord0",
	"texCoord1",
	"texCoord2",
	"texCoord3",
	"tangent",
	"bitangent",
	"triangleID",
	"custom0",
	"custom1",
	"custom2",
	"custom3",
	"custom4",
	"custom5",
	"index"};


unsigned int 
VertexData::GetAttribIndex(std::string &attribName) {

	unsigned int index;

	for (index = 0; index < VertexData::MaxAttribs; index++) {

		if (VertexData::Syntax[index].compare(attribName) == 0)
			return(index);
	}
	assert("Attrib Index out of range");
	return (MaxAttribs);
}


std::shared_ptr<VertexData>
VertexData::Create (const std::string &name) {

#ifdef NAU_OPENGL
	return std::shared_ptr<VertexData>(new GLVertexArray(name));
#elif NAU_DIRECTX
	return  std::shared_ptr<VertexData>(new DXVertexArray(name));
#endif
}


VertexData::VertexData(void) {

}


VertexData::VertexData(const std::string &name): m_Name(name) {

}


VertexData::~VertexData(void) {

}


void
VertexData::setName(std::string &name) {

	m_Name = name;
}


std::shared_ptr<std::vector<VertexData::Attr>> &
VertexData::getDataOf (unsigned int type) {

	return m_InternalArrays[type];
}


void 
VertexData::setDataFor (unsigned index, std::shared_ptr<std::vector<Attr>> &dataArray) {

	assert(index < VertexData::MaxAttribs && dataArray);

	if (!m_InternalArrays[index])
		m_InternalArrays[index].reset();

	if (index < VertexData::MaxAttribs && dataArray) {
		m_InternalArrays[index] = dataArray;
	}
}


int
VertexData::add (std::shared_ptr<VertexData> &aVertexData) {

	size_t offset = 0;
	std::string s = "position";
	std::shared_ptr<std::vector<Attr>> &vertices = getDataOf (GetAttribIndex(s));

	offset = vertices->size();

	if (offset == 0) {

		for (int i = 0; i < VertexData::MaxAttribs; i++) {
			std::shared_ptr<std::vector<Attr>> &newVec = aVertexData->getDataOf(i);
			if (!newVec) {
				std::shared_ptr<std::vector<Attr>> aVec = std::shared_ptr<std::vector<Attr>>(new std::vector<Attr>);
				aVec->insert(aVec->end(),newVec->begin(), newVec->end());
				setDataFor(i,aVec);
			}
		}
	}
	else {

		for (int i = 0; i < VertexData::MaxAttribs; i++) {

			std::shared_ptr<std::vector<Attr>> &thisVec = getDataOf(i);
			std::shared_ptr<std::vector<Attr>> &newVec = aVertexData->getDataOf(i);

			if (newVec && thisVec) {
			
				thisVec->insert(thisVec->end(),newVec->begin(), newVec->end());
			}
			else if (newVec && !thisVec) {
			
				std::shared_ptr<std::vector<Attr>> aVec = std::shared_ptr<std::vector<Attr>>(new std::vector<Attr>(offset));
				aVec->insert(aVec->end(), newVec->begin(), newVec->end());
				setDataFor(i,aVec);
			}
			else if (!newVec && thisVec) {

				size_t size = aVertexData->getDataOf(GetAttribIndex(std::string("position")))->size();
				thisVec->resize(offset + size);
			}
		}
	}
	return (int)offset;
}


void 
VertexData::unitize(vec3 &vCenter, vec3 &vMin, vec3 &vMax) {

	std::shared_ptr<std::vector<Attr>> &vertices = getDataOf (GetAttribIndex(std::string("position")));

	if (!vertices) 
		return;

	float max;
	vec3 diff;

	diff = vMax;
	diff -= vMin;
	if (diff.x > diff.y) {

		if (diff.x > diff.z) {
			max = diff.x;
		}
		else {
			max = diff.z;
		} 
	}
	else if (diff.y > diff.z) {

		max = diff.y;
	}
	else {
		max = diff.z;
	}

	max *= 0.5;
	//scale vertices so that min = -1 and max = 1
	for(unsigned int i = 0; i < vertices->size(); i++) {
	
		(*vertices)[i].x = ((*vertices)[i].x - vCenter.x) / max; 
		(*vertices)[i].y = ((*vertices)[i].y - vCenter.y) / max; 
		(*vertices)[i].z = ((*vertices)[i].z - vCenter.z) / max; 
	}
}


