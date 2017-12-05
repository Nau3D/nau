#include "nau/geometry/grid.h"

#include "nau.h"
#include "nau/math/vec3.h"
#include "nau/geometry/mesh.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"
#include "nau.h"

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::material;

bool
Grid::Init() {

	//UINT
	Attribs.add(Attribute(DIVISIONS, "DIVISIONS", Enums::UINT, false, new NauUInt(10), new NauUInt(1), NULL));
	Attribs.add(Attribute(LENGTH, "LENGTH", Enums::FLOAT, false, new NauFloat(1), NULL, NULL));

	//#ifndef _WINDLL
	NAU->registerAttributes("GRID", &Attribs);
	//#endif
	return true;
}


AttribSet Grid::Attribs;
bool Grid::Inited = Init();


Grid::Grid() : Primitive(), m_Built(false) {

	registerAndInitArrays(Attribs);
}


Grid::~Grid(void) {

}


std::string 
Grid::getClassName() {

	return "Grid";
}


void
Grid::build() {

	int divs = m_UIntProps[DIVISIONS] + 1;// (int)m_Floats[SLICES] + 1;
	float length = m_FloatProps[LENGTH];
	int total = (divs) * (divs);
	std::shared_ptr<std::vector<VertexData::Attr>> vertices =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(total));
	std::shared_ptr<std::vector<VertexData::Attr>> tangents =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(total));
	std::shared_ptr<std::vector<VertexData::Attr>> textureCoords =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(total));
	std::shared_ptr<std::vector<VertexData::Attr>> normals =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(total));

	float step = length / (divs-1);
	float start = -length * 0.5f;

	for (int i = 0; i < divs; ++i) {
		for (int j = 0; j < divs; ++j) {
			vertices->at(i * (divs)+j).set(start + i*step, 0.0f, -start - j*step,1.0f);
			tangents->at(i * (divs)+j).set(1.0f, 0.0f, 0.0f, 0.0f);
			normals->at(i * (divs)+j).set(0.0f, 1.0f, 0.0f, 0.0f);
			textureCoords->at(i * (divs)+j).set(i*1.0f/ m_UIntProps[DIVISIONS], j*1.0f/ m_UIntProps[DIVISIONS], 0.0f, 0.0f);
		}
	}
	std::shared_ptr<VertexData> &vertexData = getVertexData();

	vertexData->setDataFor(VertexData::GetAttribIndex(std::string("position")), vertices);
	vertexData->setDataFor(VertexData::GetAttribIndex(std::string("tangent")), tangents);
	vertexData->setDataFor(VertexData::GetAttribIndex(std::string("texCoord0")), textureCoords);
	vertexData->setDataFor(VertexData::GetAttribIndex(std::string("normal")), normals);


	std::shared_ptr<MaterialGroup> aMaterialGroup = MaterialGroup::Create(this, "__Light Grey");

	std::shared_ptr<std::vector<unsigned int>> indices =
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>((divs - 1)*(divs - 1) * 2 * 3));

	int k = 0;
	for (int i = 0; i < divs - 1; ++i) {
		for (int j = 0; j < divs - 1; ++j) {
			indices->at(k++) = i * divs + j;
			indices->at(k++) = (i + 1) * divs + j;
			indices->at(k++) = i * divs + j + 1;

			indices->at(k++) = i * divs + j + 1;
			indices->at(k++) = (i + 1) * divs + j;
			indices->at(k++) = (i + 1) * divs + j + 1;
		}

	}
	aMaterialGroup->setIndexList(indices);
	addMaterialGroup(aMaterialGroup);

	m_Built = true;
}


void
Grid::setPropui(UIntProperty prop, int unsigned value) {

	m_UIntProps[prop] = value;
	if (m_Built)
		rebuild();
}


void
Grid::setPropf(FloatProperty prop, float value) {

	m_FloatProps[prop] = value;
	if (m_Built)
		rebuild();
}


void
Grid::rebuild() {

	int divs = m_UIntProps[DIVISIONS] + 1;// (int)m_Floats[SLICES] + 1;
	float length = m_FloatProps[LENGTH];
	int total = (divs) * (divs);
	std::shared_ptr<std::vector<VertexData::Attr>> &vertices = getVertexData()->getDataOf(VertexData::GetAttribIndex(std::string("position")));
	vertices->resize(total);
	std::shared_ptr<std::vector<VertexData::Attr>> &tangents = getVertexData()->getDataOf(VertexData::GetAttribIndex(std::string("tangent")));
	tangents->resize(total);
	std::shared_ptr<std::vector<VertexData::Attr>> textureCoords = getVertexData()->getDataOf(VertexData::GetAttribIndex(std::string("texCoord0")));
	textureCoords->resize(total);
	std::shared_ptr<std::vector<VertexData::Attr>> &normals = getVertexData()->getDataOf(VertexData::GetAttribIndex(std::string("normal")));
	normals->resize(total);

	float step = length / divs;
	float start = -length * 0.5f;

	for (int i = 0; i < divs; ++i) {
		for (int j = 0; j < divs; ++j) {
			vertices->at(i * (divs)+j).set(start + i*step, 0.0f, start + j*step, 1.0f);
			tangents->at(i * (divs)+j).set(1.0f, 0.0f, 0.0f, 0.0f);
			normals->at(i * (divs)+j).set(0.0f, 1.0f, 0.0f, 0.0f);
			textureCoords->at(i * (divs)+j).set(i*1.0f / divs, j*1.0f / divs, 0.0f, 0.0f);
		}
	}

	std::shared_ptr<MaterialGroup> aMaterialGroup = getMaterialGroups()[0];

	std::shared_ptr<std::vector<unsigned int>> &indices = aMaterialGroup->getIndexData()->getIndexData();
	indices->resize((divs - 1)*(divs - 1) * 2 * 3);

	int k = 0;
	for (int i = 0; i < divs - 1; ++i) {
		for (int j = 0; j < divs - 1; ++j) {
			indices->at(k++) = i * divs + j;
			indices->at(k++) = (i + 1) * divs + j;
			indices->at(k++) = i * divs + j + 1;

			indices->at(k++) = i * divs + j + 1;
			indices->at(k++) = (i + 1) * divs + j;
			indices->at(k++) = (i + 1) * divs + j + 1;
		}

	}
	aMaterialGroup->resetCompilationFlag();
	aMaterialGroup->compile();
}

