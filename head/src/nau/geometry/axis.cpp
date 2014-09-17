#include <nau/geometry/axis.h>

#include <nau/math/vec3.h>
#include <nau/render/vertexdata.h>
#include <nau/material/materialgroup.h>

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::material;


Axis::Axis(void) : Primitive()
	
{
	setDrawingPrimitive(nau::render::IRenderable::LINES);
	std::vector<VertexData::Attr> *vertices = new std::vector<vec4>(6);
	std::vector<VertexData::Attr> *normals = new std::vector<vec4>(6);

	// FRONT
	vertices->at (0).set	(-1.0f,  0.0f,  0.0f);
	vertices->at (1).set	( 1.0f,  0.0f,  0.0f);
	vertices->at (2).set	( 0.0f, -1.0f,  0.0f);
	vertices->at (3).set	( 0.0f,  1.0f,  0.0f);
	vertices->at (4).set	( 0.0f,  0.0f, -1.0f);
	vertices->at (5).set	( 0.0f,  0.0f,  1.0f);

	VertexData &vertexData = getVertexData();

	vertexData.setDataFor (VertexData::getAttribIndex("position"), vertices);
	vertexData.setDataFor (VertexData::getAttribIndex("normal"), normals);

	MaterialGroup *aMaterialGroup = new MaterialGroup();
	std::vector<unsigned int> *indices = new std::vector<unsigned int>(2);
	indices->at (0) = 0;		
	indices->at (1) = 1;
	aMaterialGroup->setIndexList (indices);
	aMaterialGroup->setParent (this);
	aMaterialGroup->setMaterialName("__Emission Red");
	addMaterialGroup (aMaterialGroup);

	aMaterialGroup = new MaterialGroup();
	indices = new std::vector<unsigned int>(2);
	indices->at (0) = 2;		
	indices->at (1) = 3;
	aMaterialGroup->setIndexList (indices);
	aMaterialGroup->setParent (this);
	aMaterialGroup->setMaterialName("__Emission Green");
	addMaterialGroup (aMaterialGroup);

	aMaterialGroup = new MaterialGroup();
	indices = new std::vector<unsigned int>(2);
	indices->at (0) = 4;		
	indices->at (1) = 5;
	aMaterialGroup->setIndexList (indices);
	aMaterialGroup->setParent (this);
	aMaterialGroup->setMaterialName("__Emission Blue");
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