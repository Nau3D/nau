#include "nau/geometry/boundingBoxPrimitive.h"

#include "nau/math/vec3.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::material;



BBox::BBox(void) : Primitive() {

	float n = 1.0f;

	std::shared_ptr<std::vector<VertexData::Attr>> vertices =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(8));
	std::shared_ptr<std::vector<VertexData::Attr>> normals =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(8));

	//BOTTOM
	vertices->at (0).set(-n, -n,  n);
	vertices->at (1).set( n, -n,  n);
	vertices->at (2).set( n, -n, -n);
	vertices->at (3).set(-n, -n, -n);

	//TOP
	vertices->at (4).set(-n,  n,  n);
	vertices->at (5).set( n,  n,  n);
	vertices->at (6).set( n,  n, -n);
	vertices->at (7).set(-n,  n, -n);

	std::shared_ptr<VertexData> &vertexData = getVertexData();
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("position")), vertices);
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("normal")), normals);

	//FRONT
	std::shared_ptr<std::vector<unsigned int>> indices = 
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>(4));
	indices->at (0) = 0;		
	indices->at (1) = 1;
	indices->at (2) = 5;
	indices->at (3) = 4;

	std::shared_ptr<MaterialGroup> aMaterialGroup = MaterialGroup::Create(this, "__nau_material_lib::__Emission Blue");
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);

	//LEFT
	indices.reset(new std::vector<unsigned int>(4));
	indices->at (0) = 0;		
	indices->at (1) = 4;
	indices->at (2) = 7;
	indices->at (3) = 3;

	aMaterialGroup.reset();
	aMaterialGroup = MaterialGroup::Create(this, "__nau_material_lib::__Emission Cyan");
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);

	//BACK
	indices.reset(new std::vector<unsigned int>(4));
	indices->at (0)= 2;		
	indices->at (1)= 3;
	indices->at (2)= 7;
	indices->at (3)= 6;

	aMaterialGroup.reset();
	aMaterialGroup = MaterialGroup::Create(this, "__nau_material_lib::__Emission Yellow");
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);

	//RIGHT
	indices.reset(new std::vector<unsigned int>(4));
	indices->at (0)= 1;		
	indices->at (1)= 2;
	indices->at (2)= 6;
	indices->at (3)= 5;

	aMaterialGroup.reset();
	aMaterialGroup = MaterialGroup::Create(this, "__nau_material_lib::__Emission Red");
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);

	//TOP
	indices.reset(new std::vector<unsigned int>(4));
	indices->at (0)= 4;		
	indices->at (1)= 5;
	indices->at (2)= 6;
	indices->at (3)= 7;

	aMaterialGroup.reset();
	aMaterialGroup = MaterialGroup::Create(this, "__nau_material_lib::__Emission Green");
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);

	//BOTTOM
	indices.reset(new std::vector<unsigned int>(4));
	indices->at (0)= 0;		
	indices->at (1)= 1;
	indices->at (2)= 2;
	indices->at (3)= 3;

	aMaterialGroup.reset();
	aMaterialGroup = MaterialGroup::Create(this, "__nau_material_lib::__Emission Purple");
	aMaterialGroup->setIndexList (indices);
	addMaterialGroup (aMaterialGroup);
}


BBox::~BBox(void) {

}


void 
BBox::build() {

}

