#include "nau/geometry/sphere.h"

#include "nau.h"
#include "nau/math/vec3.h"
#include "nau/render/vertexdata.h"
#include "nau/material/materialgroup.h"

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::material;


const std::string Sphere::FloatParamNames[] = {"slices", "stacks"};



bool
Sphere::InitSphere() {

	//UINT
	Attribs.add(Attribute(STACKS, "STACKS", Enums::UINT, false, new unsigned int(10), new unsigned int(2), NULL));
	Attribs.add(Attribute(SLICES, "SLICES", Enums::UINT, false, new unsigned int(10), new unsigned int(3), NULL));

	return true;
}


AttribSet Sphere::Attribs;
bool Sphere::InitedSphere = InitSphere();


Sphere::Sphere(): Primitive()/*,
	m_Floats(COUNT_FLOATPARAMS)*/ {

		registerAndInitArrays(Attribs);
}


Sphere::~Sphere(void)
{

}


void 
Sphere::build() {

	int slices = m_UIntProps[SLICES] + 1;// (int)m_Floats[SLICES] + 1;
	int stacks = m_UIntProps[STACKS] + 1;//(int)m_Floats[STACKS] + 1;
	int total = (slices) * (stacks);
	std::vector<VertexData::Attr> *vertices = new std::vector<vec4>(total);
	std::vector<VertexData::Attr> *tangents = new std::vector<vec4>(total);
	std::vector<VertexData::Attr> *textureCoords = new std::vector<vec4>(total);
	std::vector<VertexData::Attr> *normals = new std::vector<vec4>(total);

	float stepSlice = 2.0f * M_PI / (slices-1);
	float stepStack = M_PI / (stacks-1);

	for (int i = 0; i < stacks; ++i) {
		for (int j = 0; j < slices; ++j) {
			float cosAlpha = cos(j * stepSlice);
			float sinAlpha = sin(j * stepSlice);
			float sinBeta = sin(i * stepStack - M_PI * 0.5);
			float cosBeta = cos(i * stepStack - M_PI * 0.5);
			vertices->at(i * (slices) + j).set(cosAlpha*cosBeta, sinBeta, sinAlpha*cosBeta);
			tangents->at(i * (slices) + j).set(cosAlpha*sinBeta, cosBeta, sinAlpha*sinBeta);
			normals->at(i * (slices) + j).set(cosAlpha*cosBeta, sinBeta, sinAlpha*cosBeta);
			textureCoords->at(i * (slices) + j).set(j*1.0f/(stacks-1),i*1.0f/(slices-1), 0.0f);
		}
	}
	VertexData &vertexData = getVertexData();

	vertexData.setDataFor (VertexData::getAttribIndex("position"), vertices);
	vertexData.setDataFor (VertexData::getAttribIndex("tangent"), tangents);
	vertexData.setDataFor (VertexData::getAttribIndex("texCoord0"), textureCoords);
	vertexData.setDataFor (VertexData::getAttribIndex("normal"), normals);


	MaterialGroup *aMaterialGroup = MaterialGroup::Create(this, "__Light Grey");
	
	std::vector<unsigned int> *indices = new std::vector<unsigned int>((slices)*(stacks)*2*3);

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
	//aMaterialGroup->setParent (this);
	//aMaterialGroup->setMaterialName("Light Grey");

	addMaterialGroup (aMaterialGroup);
}


const std::string &
Sphere::getParamfName(unsigned int i) 
{
	if (i < Sphere::COUNT_FLOATPARAMS)
		return Sphere::FloatParamNames[i];
	else
		return Primitive::NoParam;
}


float 
Sphere::getParamf(unsigned int param)
{
	assert(param < Sphere::COUNT_FLOATPARAMS);

	if (param < Sphere::COUNT_FLOATPARAMS)
		return(m_Floats[param]);
	else
		return (0.0f);
}


void
Sphere::setParam(unsigned int param, float value)
{
	assert(param < Sphere::COUNT_FLOATPARAMS);

	if (param < Sphere::COUNT_FLOATPARAMS)
		m_Floats[param] = value;
}


unsigned int
Sphere::translate(const std::string &name) 
{
	for (int i = 0; i < Sphere::COUNT_FLOATPARAMS; ++i) {
		if (FloatParamNames[i] == name)
			return i;
	}
	assert("name is not a primitive param");
	return (0);
}