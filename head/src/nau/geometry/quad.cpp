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
	std::shared_ptr<std::vector<VertexData::Attr>> vertices = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(4));
	std::shared_ptr<std::vector<VertexData::Attr>> textureCoords = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(4));
	std::shared_ptr<std::vector<VertexData::Attr>> normals = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(4));

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

	std::shared_ptr<VertexData> &vertexData = renderable->getVertexData();

	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("position")), vertices);
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("texCoord0")), textureCoords);
	vertexData->setDataFor (VertexData::GetAttribIndex(std::string("normal")), normals);

	std::shared_ptr<MaterialGroup> aMaterialGroup = MaterialGroup::Create(renderable, "__Quad");
	std::shared_ptr<std::vector<unsigned int>> indices =
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>(6));
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


Quad::~Quad(void) {

}


void
Quad::eventReceived(const std::string &sender,
	const std::string &eventType,
	const std::shared_ptr<IEventData> &evt) {


}




