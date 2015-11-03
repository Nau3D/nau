#include "nau/geometry/quad.h"

#include "nau/math/vec3.h"
#include "nau/geometry/mesh.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"
#include "nau.h"

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::material;

Quad::Quad(void) : 
	SceneObject()
{
	Mesh *renderable = (Mesh *)RESOURCEMANAGER->createRenderable("Mesh");//new Mesh;
	std::vector<VertexData::Attr> *vertices = new std::vector<vec4>(4);
	std::vector<VertexData::Attr> *textureCoords = new std::vector<vec4>(4);
	std::vector<VertexData::Attr> *normals = new std::vector<vec4>(4);

	for (int i = 0; i < 4; ++i)
		normals->at(i).set(0.0f, 0.0f, 0.0f);

	vertices->at (0).set (-1.0f, -1.0f, -1.0f);
	vertices->at (1).set (1.0f, -1.0f, -1.0f);
	vertices->at (2).set (1.0f, 1.0f, -1.0f);
	vertices->at (3).set (-1.0f, 1.0f, -1.0f);

	textureCoords->at (0).set (0.0f, 0.0f, 0.0f);
	textureCoords->at (1).set (1.0f, 0.0f, 0.0f);
	textureCoords->at (2).set (1.0f, 1.0f, 0.0f);
	textureCoords->at (3).set (0.0f, 1.0f, 0.0f);

	VertexData &vertexData = renderable->getVertexData();

	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("position")), vertices);
	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("texCoord0")), textureCoords);
	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("normal")), normals);

	MaterialGroup *aMaterialGroup = MaterialGroup::Create(renderable, "__Quad");
	

	std::vector<unsigned int> *indices = new std::vector<unsigned int>(6);


	indices->at (0) = 0;
	indices->at (1) = 1;
	indices->at (2) = 3;
	indices->at (3) = 1;
	indices->at (4) = 2;
	indices->at (5) = 3;

	aMaterialGroup->setIndexList (indices);

	renderable->addMaterialGroup (aMaterialGroup);
	setRenderable (renderable);
}

Quad::~Quad(void)
{
	delete m_Renderable;
}
