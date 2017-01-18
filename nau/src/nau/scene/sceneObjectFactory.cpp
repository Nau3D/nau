#include "nau/scene/sceneObjectFactory.h"

#include "nau/scene/sceneObject.h"
#include "nau/scene/octreeNode.h"
#include "nau/scene/geometricObject.h"
#include "nau.h"

using namespace nau::scene;


std::shared_ptr<SceneObject>
SceneObjectFactory::Create(const std::string &type)
{
	if (0 == type.compare("SimpleObject")) {
		return std::shared_ptr<SceneObject>(new SceneObject);
	}

	else if (0 == type.compare("OctreeNode")) {
		return std::shared_ptr<SceneObject>(new OctreeNode);
	}

	else if (0 == type.compare("Geometry")) {
		return std::shared_ptr<SceneObject>(new GeometricObject);
	}
	else {
		assert("SceneObjectFactory: type is not valid");
		return NULL;
	}
}