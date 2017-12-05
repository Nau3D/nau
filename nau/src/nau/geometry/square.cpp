#include "nau/geometry/square.h"

#include "nau/math/vec3.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"

using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;
using namespace nau::math;


Square::Square(void) : Primitive() {

	float n = 1.0f;

	std::shared_ptr<std::vector<VertexData::Attr>> vertices = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(4));
	std::shared_ptr<std::vector<VertexData::Attr>> tangent = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(4));
	std::shared_ptr<std::vector<VertexData::Attr>> textureCoords = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(4));
	std::shared_ptr<std::vector<VertexData::Attr>> normals = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(4));

	//BOTTOM
	vertices->at (Square::TOP_LEFT).set		(-n, 0.0f,  n);
	vertices->at (Square::TOP_RIGHT).set		( n, 0.0f,  n);
	vertices->at (Square::BOTTOM_RIGHT).set	( n, 0.0f,  -n);
	vertices->at (Square::BOTTOM_LEFT).set	(-n, 0.0f,  -n);

	tangent->at (Square::TOP_LEFT).set		(0.0f,  0.0f, -1.0f);
	tangent->at (Square::TOP_RIGHT).set		(0.0f,  0.0f, -1.0f);
	tangent->at (Square::BOTTOM_RIGHT).set	(0.0f,  0.0f, -1.0f);
	tangent->at (Square::BOTTOM_LEFT).set	(0.0f,  0.0f, -1.0f);

	textureCoords->at (Square::TOP_LEFT).set		(0.0f, n,    0.0f);
	textureCoords->at (Square::TOP_RIGHT).set	(n,    n,    0.0f);
	textureCoords->at (Square::BOTTOM_RIGHT).set	(n,    0.0f, 0.0f);
	textureCoords->at (Square::BOTTOM_LEFT).set	(0.0f, 0.0f, 0.0f);

	normals->at (Square::TOP_LEFT).set		( 0.0f, 1.0f, 0.0f);
	normals->at (Square::TOP_RIGHT).set		( 0.0f, 1.0f, 0.0f);
	normals->at (Square::BOTTOM_RIGHT).set	( 0.0f, 1.0f, 0.0f);	
	normals->at (Square::BOTTOM_LEFT).set	( 0.0f, 1.0f, 0.0f); 	

	std::shared_ptr<VertexData> &vertexData = getVertexData();

	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("position")), vertices);
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("texCoord0")), textureCoords);
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("tangent")), tangent);
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("normal")), normals);

	std::shared_ptr<MaterialGroup> aMaterialGroup = MaterialGroup::Create(this, "__Light Grey");
	
	std::shared_ptr<std::vector<unsigned int>> indices = 
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>(6));

	//BOTTOM
	indices->at (0)= Square::BOTTOM_LEFT;
	indices->at (1)= Square::TOP_LEFT;		
	indices->at (2)= Square::TOP_RIGHT;

	indices->at (3)= Square::BOTTOM_RIGHT;
	indices->at (4)= Square::BOTTOM_LEFT;
	indices->at (5)= Square::TOP_RIGHT;

	aMaterialGroup->setIndexList (indices);

	addMaterialGroup (aMaterialGroup);
}


Square::~Square(void) {

}


std::string
Square::getClassName() {

	return "Square";
}


void
Square::build() {

}


