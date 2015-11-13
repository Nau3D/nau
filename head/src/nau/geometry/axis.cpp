#include "nau/geometry/axis.h"

#include "nau/math/vec3.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::material;



Axis::Axis(void) : Primitive()
	
{
	setDrawingPrimitive(nau::render::IRenderable::LINES);
	std::vector<VertexData::Attr> *vertices = new std::vector<VertexData::Attr>(6);
	//std::vector<VertexData::Attr> *normals = new std::vector<vec4>(6);

	// FRONT
	vertices->at (0).set	(-1.0f,  0.0f,  0.0f);
	vertices->at (1).set	( 1.0f,  0.0f,  0.0f);
	vertices->at (2).set	( 0.0f, -1.0f,  0.0f);
	vertices->at (3).set	( 0.0f,  1.0f,  0.0f);
	vertices->at (4).set	( 0.0f,  0.0f, -1.0f);
	vertices->at (5).set	( 0.0f,  0.0f,  1.0f);

	VertexData &vertexData = getVertexData();

	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("position")), vertices);
	//vertexData.setDataFor (VertexData::getAttribIndex("normal"), normals);

	MaterialGroup *aMaterialGroup = MaterialGroup::Create(this, "__Emission Red");
	std::vector<unsigned int> *indices = new std::vector<unsigned int>(2);
	indices->at (0) = 0;		
	indices->at (1) = 1;
	aMaterialGroup->setIndexList (indices);
	//aMaterialGroup->setParent (this);
	//aMaterialGroup->setMaterialName("__Emission Red");
	addMaterialGroup (aMaterialGroup);

	aMaterialGroup = MaterialGroup::Create(this, "__Emission Green");
	indices = new std::vector<unsigned int>(2);
	indices->at (0) = 2;		
	indices->at (1) = 3;
	aMaterialGroup->setIndexList (indices);
	//aMaterialGroup->setParent (this);
	//aMaterialGroup->setMaterialName("__Emission Green");
	addMaterialGroup (aMaterialGroup);

	aMaterialGroup = MaterialGroup::Create(this, "__Emission Blue");
	indices = new std::vector<unsigned int>(2);
	indices->at (0) = 4;		
	indices->at (1) = 5;
	aMaterialGroup->setIndexList (indices);
	//aMaterialGroup->setParent (this);
	//aMaterialGroup->setMaterialName("__Emission Blue");
	addMaterialGroup (aMaterialGroup);
}


Axis::~Axis(void)
{

}


void 
Axis::build()
{
}


unsigned int
Axis::translate(const std::string &name) 
{
	assert("name is not a primitive param");
	return (0);
}