#include "nau/geometry/axis.h"

#include "nau/math/vec3.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::material;



Axis::Axis(void) : Primitive() {

	setDrawingPrimitive(nau::render::IRenderable::LINES);
	std::vector<VertexData::Attr> *vertices = new std::vector<VertexData::Attr>(6);

	vertices->at (0).set	(-1.0f,  0.0f,  0.0f);
	vertices->at (1).set	( 1.0f,  0.0f,  0.0f);
	vertices->at (2).set	( 0.0f, -1.0f,  0.0f);
	vertices->at (3).set	( 0.0f,  1.0f,  0.0f);
	vertices->at (4).set	( 0.0f,  0.0f, -1.0f);
	vertices->at (5).set	( 0.0f,  0.0f,  1.0f);

	VertexData &vertexData = getVertexData();

	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("position")), vertices);

	std::shared_ptr<MaterialGroup> aMaterialGroup = MaterialGroup::Create(this, "__Emission Red");
	std::shared_ptr<std::vector<unsigned int>> indices =
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>(2));
	indices->at (0) = 0;
	indices->at (1) = 1;
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);

	aMaterialGroup.reset();
	aMaterialGroup = MaterialGroup::Create(this, "__Emission Green");
	indices.reset(new std::vector<unsigned int>(2));
	indices->at (0) = 2;		
	indices->at (1) = 3;
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);

	aMaterialGroup.reset();
	aMaterialGroup = MaterialGroup::Create(this, "__Emission Blue");
	indices.reset(new std::vector<unsigned int>(2));
	indices->at (0) = 4;		
	indices->at (1) = 5;
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);
}


Axis::~Axis(void) {

}


void 
Axis::build() {

}


