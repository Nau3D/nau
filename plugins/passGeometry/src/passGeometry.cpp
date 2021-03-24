#include "passGeometry.h"

#include "iNau.h"
#include "nau.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/iTexture.h"
#include "nau/material/materialGroup.h"
#include "nau/math/vec3.h"
#include "nau/render/passFactory.h"
#include "nau/scene/sceneFactory.h"



#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>

#ifdef WIN32
#include <Windows.h>
#endif

static char className[] = "geometryPI";

#ifdef WIN32
#define EXPORT __declspec(dllexport)
#elif __linux__
#define EXPORT extern "C"
#endif

EXPORT 
void *
createPass(const char *s) {

	std::shared_ptr<PassGeometry> *p = new std::shared_ptr<PassGeometry>(new PassGeometry(s));
	return p;
}


EXPORT
void 
init(void *nauInst) {

	INau::SetInterface((nau::INau *)nauInst);
	nau::Nau::SetInstance((nau::Nau *)nauInst);
	glbinding::Binding::initialize(false);
}


EXPORT
char *
getClassName() {

	return className;
}


using namespace nau::geometry;
using namespace nau::math;
using namespace nau::render;
using namespace nau::scene;

Pass *
PassGeometry::Create(const std::string &passName) {

	return new PassGeometry(passName);
}


PassGeometry::PassGeometry(const std::string &passName) :
	Pass (passName) {

	m_ClassName = "geometryPI";
	m_Inited = false;
}


PassGeometry::~PassGeometry(void) {

}


void 
PassGeometry::prepareGeometry() {

	std::shared_ptr<IScene> m_Scene = RENDERMANAGER->createScene("my_scene", "Scene");

	// create a renderable
	std::shared_ptr<nau::render::IRenderable> &m_Renderable = RESOURCEMANAGER->createRenderable("Mesh", "my_plane");

	// fill in vertex array
	int vertexCount = 4;

	std::shared_ptr<std::vector<VertexData::Attr>> vertices =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(vertexCount));

	vertices->at(0).set(-1, 0, -1);
	vertices->at(1).set(-1, 0,  1);
	vertices->at(2).set( 1, 0,  1);
	vertices->at(3).set( 1, 0, -1);

	std::shared_ptr<VertexData>& vertexData = m_Renderable->getVertexData();

	vertexData->setDataFor(VertexData::GetAttribIndex(std::string("position")), vertices);

	// create indices and fill the array
	int indexCount = 6;
	std::shared_ptr<std::vector<unsigned int>> indices =
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>(indexCount));

	indices->at(0) = 0;
	indices->at(1) = 1;
	indices->at(2) = 2;

	indices->at(3) = 0;
	indices->at(4) = 2;
	indices->at(5) = 3;


	// create the material group
	std::shared_ptr<MaterialGroup> aMaterialGroup = MaterialGroup::Create(m_Renderable.get(), "__Light Grey");
	aMaterialGroup->setIndexList(indices);
	m_Renderable->addMaterialGroup(aMaterialGroup);

	std::shared_ptr<SceneObject> &m_SceneObject = nau::scene::SceneObjectFactory::Create("SimpleObject");

	m_SceneObject->setRenderable(m_Renderable);
	
	m_Scene->add(m_SceneObject);

	addScene("my_scene");

	m_Inited = true;


}


void
PassGeometry::prepare (void) {

	if (!m_Inited) {

		prepareGeometry();
	}
	prepareBuffers();
	setupCamera();

}


void
PassGeometry::restore (void) {

}


void 
PassGeometry::doPass (void) {

	std::vector<std::string>::iterator scenesIter;
	scenesIter = m_SceneVector.begin();

	for (; scenesIter != m_SceneVector.end(); ++scenesIter) {
		std::shared_ptr<IScene>& aScene = RENDERMANAGER->getScene(*scenesIter);
		std::vector<std::shared_ptr<SceneObject>> sceneObjects;
		aScene->getAllObjects(&sceneObjects);
		std::vector<SceneObject*>::iterator objIter;

		for (auto& so : sceneObjects) {

			RENDERMANAGER->addToQueue(so, m_MaterialMap);
		}
	}

	RENDERER->setDepthClamping(true);

	RENDERMANAGER->processQueue();

}

