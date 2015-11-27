#include "nau/geometry/sphere.h"

#include "nau.h"
#include "nau/math/vec3.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::material;


bool
Sphere::Init() {

	//UINT
	Attribs.add(Attribute(STACKS, "STACKS", Enums::UINT, false, new NauUInt(10), new NauUInt(2), NULL));
	Attribs.add(Attribute(SLICES, "SLICES", Enums::UINT, false, new NauUInt(10), new NauUInt(3), NULL));

	NAU->registerAttributes("SPHERE", &Attribs);
	return true;
}


AttribSet Sphere::Attribs;
bool Sphere::Inited = Init();


Sphere::Sphere(): Primitive() {

		registerAndInitArrays(Attribs);
}


Sphere::~Sphere(void) {

}


void 
Sphere::build() {

	int slices = m_UIntProps[SLICES] + 1;// (int)m_Floats[SLICES] + 1;
	int stacks = m_UIntProps[STACKS] + 1;//(int)m_Floats[STACKS] + 1;
	int total = (slices) * (stacks);
	std::shared_ptr<std::vector<VertexData::Attr>> vertices = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(total));
	std::shared_ptr<std::vector<VertexData::Attr>> tangents = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(total));
	std::shared_ptr<std::vector<VertexData::Attr>> textureCoords = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(total));
	std::shared_ptr<std::vector<VertexData::Attr>> normals = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(total));

	float stepSlice = 2.0f * (float)M_PI / (slices-1);
	float stepStack = (float)M_PI / (stacks-1);

	for (int i = 0; i < stacks; ++i) {
		for (int j = 0; j < slices; ++j) {
			float cosAlpha = cos(j * stepSlice);
			float sinAlpha = sin(j * stepSlice);
			float sinBeta = sin(i * stepStack - (float)M_PI * 0.5f);
			float cosBeta = cos(i * stepStack - (float)M_PI * 0.5f);
			vertices->at(i * (slices) + j).set(cosAlpha*cosBeta, sinBeta, sinAlpha*cosBeta);
			tangents->at(i * (slices) + j).set(cosAlpha*sinBeta, cosBeta, sinAlpha*sinBeta);
			normals->at(i * (slices) + j).set(cosAlpha*cosBeta, sinBeta, sinAlpha*cosBeta);
			textureCoords->at(i * (slices) + j).set(j*1.0f/(stacks-1),i*1.0f/(slices-1), 0.0f);
		}
	}
	std::shared_ptr<VertexData> &vertexData = getVertexData();

	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("position")), vertices);
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("tangent")), tangents);
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("texCoord0")), textureCoords);
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("normal")), normals);


	std::shared_ptr<MaterialGroup> aMaterialGroup = MaterialGroup::Create(this, "__Light Grey");
	
	std::shared_ptr<std::vector<unsigned int>> indices = 
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>((slices)*(stacks)*2*3));

	int k =  0;
	for (int i = 0; i < stacks-1; ++i) {
		for (int j = 0; j < slices-1; ++j) {
			indices->at(k++) = i * slices + j;
			indices->at(k++)   = (i+1) * slices + j;
			indices->at(k++) = i * slices + j + 1;

			indices->at(k++) = i * slices + j + 1;
			indices->at(k++) = (i+1) * slices + j;
			indices->at(k++) = (i+1) * slices + j + 1;
		}
	
	}
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);
}


