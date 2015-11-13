#include "nau/geometry/box.h"

#include "nau/math/vec3.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::material;


const std::string Box::FloatParamNames[] = {""};

Box::Box(void) : Primitive(),
	m_Floats(COUNT_FLOATPARAMS)
{
	float n = 1.0f;

	std::vector<VertexData::Attr> *vertices = new std::vector<VertexData::Attr>(24);
	std::vector<VertexData::Attr> *tangent = new std::vector<VertexData::Attr>(24);
	std::vector<VertexData::Attr> *textureCoords = new std::vector<VertexData::Attr>(24);
	std::vector<VertexData::Attr> *normals = new std::vector<VertexData::Attr>(24);

	// FRONT
	vertices->at (Box::FACE_FRONT + Box::TOP_LEFT).set		(-n,  n, n);
	vertices->at (Box::FACE_FRONT + Box::TOP_RIGHT).set		( n,  n, n);
	vertices->at (Box::FACE_FRONT + Box::BOTTOM_RIGHT).set	( n, -n, n);
	vertices->at (Box::FACE_FRONT + Box::BOTTOM_LEFT).set	(-n, -n, n);

	tangent->at (Box::FACE_FRONT + Box::TOP_LEFT).set		(0.0f,  1.0f, 0.0f);
	tangent->at (Box::FACE_FRONT + Box::TOP_RIGHT).set		(0.0f,  1.0f, 0.0f);
	tangent->at (Box::FACE_FRONT + Box::BOTTOM_RIGHT).set	(0.0f,  1.0f, 0.0f);
	tangent->at (Box::FACE_FRONT + Box::BOTTOM_LEFT).set	(0.0f,  1.0f, 0.0f);

	textureCoords->at (Box::FACE_FRONT + Box::TOP_LEFT).set		(0.0f, n, 0.0f);
	textureCoords->at (Box::FACE_FRONT + Box::TOP_RIGHT).set	(n, n, 0.0f);
	textureCoords->at (Box::FACE_FRONT + Box::BOTTOM_RIGHT).set	(n, 0.0f, 0.0f);
	textureCoords->at (Box::FACE_FRONT + Box::BOTTOM_LEFT).set	(0.0f, 0.0f, 0.0f);

	normals->at (Box::FACE_FRONT + Box::TOP_LEFT).set		(0.0f, 0.0f, n);
	normals->at (Box::FACE_FRONT + Box::TOP_RIGHT).set		(0.0f, 0.0f, n);
	normals->at (Box::FACE_FRONT + Box::BOTTOM_RIGHT).set	(0.0f, 0.0f, n);	
	normals->at (Box::FACE_FRONT + Box::BOTTOM_LEFT).set	(0.0f, 0.0f, n); 	

	//LEFT
	vertices->at (Box::FACE_LEFT + Box::TOP_LEFT).set		(-n, n, -n);
	vertices->at (Box::FACE_LEFT + Box::TOP_RIGHT).set		(-n, n,  n);
	vertices->at (Box::FACE_LEFT + Box::BOTTOM_RIGHT).set	(-n, -n, n);
	vertices->at (Box::FACE_LEFT + Box::BOTTOM_LEFT).set	(-n, -n, -n);

	tangent->at (Box::FACE_LEFT + Box::TOP_LEFT).set		(0.0f,  1.0f, 0.0f);(-n, n, -n);
	tangent->at (Box::FACE_LEFT + Box::TOP_RIGHT).set		(0.0f,  1.0f, 0.0f);(-n, n,  n);
	tangent->at (Box::FACE_LEFT + Box::BOTTOM_RIGHT).set	(0.0f,  1.0f, 0.0f);(-n, -n, n);
	tangent->at (Box::FACE_LEFT + Box::BOTTOM_LEFT).set		(0.0f,  1.0f, 0.0f);(-n, -n, -n);

	textureCoords->at (Box::FACE_LEFT + Box::TOP_LEFT).set		(0.0f, n, 0.0f);
	textureCoords->at (Box::FACE_LEFT + Box::TOP_RIGHT).set		(n, n, 0.0f);
	textureCoords->at (Box::FACE_LEFT + Box::BOTTOM_RIGHT).set	(n, 0.0f, 0.0f);
	textureCoords->at (Box::FACE_LEFT + Box::BOTTOM_LEFT).set	(0.0f, 0.0f, 0.0f);

	normals->at (Box::FACE_LEFT + Box::TOP_LEFT).set		(-n, 0.0f, 0.0f);
	normals->at (Box::FACE_LEFT + Box::TOP_RIGHT).set		(-n, 0.0f, 0.0f);
	normals->at (Box::FACE_LEFT + Box::BOTTOM_RIGHT).set	(-n, 0.0f, 0.0f);	
	normals->at (Box::FACE_LEFT + Box::BOTTOM_LEFT).set		(-n, 0.0f, 0.0f); 	

	//BACK
	vertices->at (Box::FACE_BACK + Box::TOP_LEFT).set		( n, n, -n);
	vertices->at (Box::FACE_BACK + Box::TOP_RIGHT).set		(-n, n, -n);
	vertices->at (Box::FACE_BACK + Box::BOTTOM_RIGHT).set	(-n, -n,-n);
	vertices->at (Box::FACE_BACK + Box::BOTTOM_LEFT).set	( n, -n,-n);

	tangent->at (Box::FACE_BACK + Box::TOP_LEFT).set		(0.0f,  1.0f, 0.0f);
	tangent->at (Box::FACE_BACK + Box::TOP_RIGHT).set		(0.0f,  1.0f, 0.0f);
	tangent->at (Box::FACE_BACK + Box::BOTTOM_RIGHT).set	(0.0f,  1.0f, 0.0f);
	tangent->at (Box::FACE_BACK + Box::BOTTOM_LEFT).set		(0.0f,  1.0f, 0.0f);

	textureCoords->at (Box::FACE_BACK + Box::TOP_LEFT).set		(0.0f, n, 0.0f);
	textureCoords->at (Box::FACE_BACK + Box::TOP_RIGHT).set		(n, n, 0.0f);
	textureCoords->at (Box::FACE_BACK + Box::BOTTOM_RIGHT).set	(n, 0.0f, 0.0f);
	textureCoords->at (Box::FACE_BACK + Box::BOTTOM_LEFT).set	(0.0f, 0.0f, 0.0f);

	normals->at (Box::FACE_BACK + Box::TOP_LEFT).set		(0.0f, 0.0f, -n);
	normals->at (Box::FACE_BACK + Box::TOP_RIGHT).set		(0.0f, 0.0f, -n);
	normals->at (Box::FACE_BACK + Box::BOTTOM_RIGHT).set	(0.0f, 0.0f, -n);	
	normals->at (Box::FACE_BACK + Box::BOTTOM_LEFT).set		(0.0f, 0.0f, -n); 	

	//RIGHT
	vertices->at (Box::FACE_RIGHT + Box::TOP_LEFT).set		( n, n,  n);
	vertices->at (Box::FACE_RIGHT + Box::TOP_RIGHT).set		( n, n, -n);
	vertices->at (Box::FACE_RIGHT + Box::BOTTOM_RIGHT).set	( n, -n,-n);
	vertices->at (Box::FACE_RIGHT + Box::BOTTOM_LEFT).set	( n, -n, n);

	tangent->at (Box::FACE_RIGHT + Box::TOP_LEFT).set		(0.0f,  1.0f, 0.0f);
	tangent->at (Box::FACE_RIGHT + Box::TOP_RIGHT).set		(0.0f,  1.0f, 0.0f);
	tangent->at (Box::FACE_RIGHT + Box::BOTTOM_RIGHT).set	(0.0f,  1.0f, 0.0f);
	tangent->at (Box::FACE_RIGHT + Box::BOTTOM_LEFT).set	(0.0f,  1.0f, 0.0f);

	textureCoords->at (Box::FACE_RIGHT + Box::TOP_LEFT).set		(0.0f, n, 0.0f);
	textureCoords->at (Box::FACE_RIGHT + Box::TOP_RIGHT).set	(n, n, 0.0f);
	textureCoords->at (Box::FACE_RIGHT + Box::BOTTOM_RIGHT).set	(n, 0.0f, 0.0f);
	textureCoords->at (Box::FACE_RIGHT + Box::BOTTOM_LEFT).set	(0.0f, 0.0f, 0.0f);

	normals->at (Box::FACE_RIGHT + Box::TOP_LEFT).set		(n, 0.0f, 0.0f);
	normals->at (Box::FACE_RIGHT + Box::TOP_RIGHT).set		(n, 0.0f, 0.0f);
	normals->at (Box::FACE_RIGHT + Box::BOTTOM_RIGHT).set	(n, 0.0f, 0.0f);	
	normals->at (Box::FACE_RIGHT + Box::BOTTOM_LEFT).set	(n, 0.0f, 0.0f); 	

	//TOP
	vertices->at (Box::FACE_TOP + Box::TOP_LEFT).set	(-n, n, -n);
	vertices->at (Box::FACE_TOP + Box::TOP_RIGHT).set	( n, n, -n);
	vertices->at (Box::FACE_TOP + Box::BOTTOM_RIGHT).set( n, n,  n);
	vertices->at (Box::FACE_TOP + Box::BOTTOM_LEFT).set	(-n, n,  n);

	tangent->at (Box::FACE_TOP + Box::TOP_LEFT).set			(0.0f,  0.0f, 1.0f);
	tangent->at (Box::FACE_TOP + Box::TOP_RIGHT).set		(0.0f,  0.0f, 1.0f);
	tangent->at (Box::FACE_TOP + Box::BOTTOM_RIGHT).set		(0.0f,  0.0f, 1.0f);
	tangent->at (Box::FACE_TOP + Box::BOTTOM_LEFT).set		(0.0f,  0.0f, 1.0f);

	textureCoords->at (Box::FACE_TOP + Box::TOP_LEFT).set		(0.0f, n, 0.0f);
	textureCoords->at (Box::FACE_TOP + Box::TOP_RIGHT).set		(n, n, 0.0f);
	textureCoords->at (Box::FACE_TOP + Box::BOTTOM_RIGHT).set	(n, 0.0f, 0.0f);
	textureCoords->at (Box::FACE_TOP + Box::BOTTOM_LEFT).set	(0.0f, 0.0f, 0.0f);

	normals->at (Box::FACE_TOP + Box::TOP_LEFT).set		( 0.0f, n, 0.0f);
	normals->at (Box::FACE_TOP + Box::TOP_RIGHT).set	( 0.0f, n, 0.0f);
	normals->at (Box::FACE_TOP + Box::BOTTOM_RIGHT).set	( 0.0f, n, 0.0f);	
	normals->at (Box::FACE_TOP + Box::BOTTOM_LEFT).set	( 0.0f, n, 0.0f); 	

	//BOTTOM
	vertices->at (Box::FACE_BOTTOM + Box::TOP_LEFT).set		(-n, -n,  n);
	vertices->at (Box::FACE_BOTTOM + Box::TOP_RIGHT).set	( n, -n,  n);
	vertices->at (Box::FACE_BOTTOM + Box::BOTTOM_RIGHT).set	( n, -n,  -n);
	vertices->at (Box::FACE_BOTTOM + Box::BOTTOM_LEFT).set	(-n, -n,  -n);

	tangent->at (Box::FACE_BOTTOM + Box::TOP_LEFT).set		(0.0f,  0.0f, -1.0f);
	tangent->at (Box::FACE_BOTTOM + Box::TOP_RIGHT).set		(0.0f,  0.0f, -1.0f);
	tangent->at (Box::FACE_BOTTOM + Box::BOTTOM_RIGHT).set	(0.0f,  0.0f, -1.0f);
	tangent->at (Box::FACE_BOTTOM + Box::BOTTOM_LEFT).set	(0.0f,  0.0f, -1.0f);

	textureCoords->at (Box::FACE_BOTTOM + Box::TOP_LEFT).set		(0.0f, n, 0.0f);
	textureCoords->at (Box::FACE_BOTTOM + Box::TOP_RIGHT).set		(n, n, 0.0f);
	textureCoords->at (Box::FACE_BOTTOM + Box::BOTTOM_RIGHT).set	(n, 0.0f, 0.0f);
	textureCoords->at (Box::FACE_BOTTOM + Box::BOTTOM_LEFT).set		(0.0f, 0.0f, 0.0f);

	normals->at (Box::FACE_BOTTOM + Box::TOP_LEFT).set		( 0.0f, -n, 0.0f);
	normals->at (Box::FACE_BOTTOM + Box::TOP_RIGHT).set		( 0.0f, -n, 0.0f);
	normals->at (Box::FACE_BOTTOM + Box::BOTTOM_RIGHT).set	( 0.0f, -n, 0.0f);	
	normals->at (Box::FACE_BOTTOM + Box::BOTTOM_LEFT).set	( 0.0f, -n, 0.0f); 	

	VertexData &vertexData = getVertexData();

	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("position")), vertices);
	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("texCoord0")), textureCoords);
	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("tangent")), tangent);
	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("normal")), normals);


	MaterialGroup *aMaterialGroup = MaterialGroup::Create(this, "Light Grey");
	
	std::vector<unsigned int> *indices = new std::vector<unsigned int>(36);
	//FRONT
	indices->at (0) = Box::FACE_FRONT + Box::TOP_LEFT;		
	indices->at (1) = Box::FACE_FRONT + Box::BOTTOM_LEFT;
	indices->at (2) = Box::FACE_FRONT + Box::TOP_RIGHT;

	indices->at (3) = Box::FACE_FRONT + Box::BOTTOM_LEFT;
	indices->at (4) = Box::FACE_FRONT + Box::BOTTOM_RIGHT;
	indices->at (5) = Box::FACE_FRONT + Box::TOP_RIGHT;

	//LEFT
	indices->at (6) = Box::FACE_LEFT + Box::TOP_LEFT;		
	indices->at (7) = Box::FACE_LEFT + Box::BOTTOM_LEFT;
	indices->at (8) = Box::FACE_LEFT + Box::TOP_RIGHT;

	indices->at (9) = Box::FACE_LEFT + Box::BOTTOM_LEFT;
	indices->at (10)= Box::FACE_LEFT + Box::BOTTOM_RIGHT;
	indices->at (11)= Box::FACE_LEFT + Box::TOP_RIGHT;

	//BACK
	indices->at (12)= Box::FACE_BACK + Box::TOP_LEFT;		
	indices->at (13)= Box::FACE_BACK + Box::BOTTOM_LEFT;
	indices->at (14)= Box::FACE_BACK + Box::TOP_RIGHT;

	indices->at (15)= Box::FACE_BACK + Box::BOTTOM_LEFT;
	indices->at (16)= Box::FACE_BACK + Box::BOTTOM_RIGHT;
	indices->at (17)= Box::FACE_BACK + Box::TOP_RIGHT;

	//RIGHT
	indices->at (18)= Box::FACE_RIGHT + Box::TOP_LEFT;		
	indices->at (19)= Box::FACE_RIGHT + Box::BOTTOM_LEFT;
	indices->at (20)= Box::FACE_RIGHT + Box::TOP_RIGHT;

	indices->at (21)= Box::FACE_RIGHT + Box::BOTTOM_LEFT;
	indices->at (22)= Box::FACE_RIGHT + Box::BOTTOM_RIGHT;
	indices->at (23)= Box::FACE_RIGHT + Box::TOP_RIGHT;

	//TOP
	indices->at (24)= Box::FACE_TOP + Box::TOP_LEFT;		
	indices->at (25)= Box::FACE_TOP + Box::BOTTOM_LEFT;
	indices->at (26)= Box::FACE_TOP + Box::TOP_RIGHT;

	indices->at (27)= Box::FACE_TOP + Box::BOTTOM_LEFT;
	indices->at (28)= Box::FACE_TOP + Box::BOTTOM_RIGHT;
	indices->at (29)= Box::FACE_TOP + Box::TOP_RIGHT;

	//BOTTOM
	indices->at (30)= Box::FACE_BOTTOM + Box::TOP_LEFT;		
	indices->at (31)= Box::FACE_BOTTOM + Box::BOTTOM_LEFT;
	indices->at (32)= Box::FACE_BOTTOM + Box::TOP_RIGHT;

	indices->at (33)= Box::FACE_BOTTOM + Box::BOTTOM_LEFT;
	indices->at (34)= Box::FACE_BOTTOM + Box::BOTTOM_RIGHT;
	indices->at (35)= Box::FACE_BOTTOM + Box::TOP_RIGHT;

	aMaterialGroup->setIndexList (indices);
	//aMaterialGroup->setParent (this);
	//aMaterialGroup->setMaterialName("Light Grey");

	addMaterialGroup (aMaterialGroup);
}


Box::~Box(void)
{

}


void 
Box::build()
{
}


const std::string &
Box::getParamfName(unsigned int i) 
{
	if (i < Box::COUNT_FLOATPARAMS)
		return Box::FloatParamNames[i];
	else
		return Primitive::NoParam;
}


float 
Box::getParamf(unsigned int param)
{
	assert(param < Box::COUNT_FLOATPARAMS);

	if (param < Box::COUNT_FLOATPARAMS)
		return(m_Floats[param]);
	else
		return (0.0f);
}


void
Box::setParam(unsigned int param, float value)
{
	assert(param < Box::COUNT_FLOATPARAMS);

	if (param < Box::COUNT_FLOATPARAMS)
		m_Floats[param] = value;
}


unsigned int
Box::translate(const std::string &name) 
{
	assert("name is not a primitive param");
	return (0);
}