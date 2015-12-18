#include "nau/scene/sceneObjectFactory.h"

#include "nau/scene/sceneObject.h"
#include "nau/scene/octreeNode.h"
#include "nau/scene/geometricObject.h"
#include "nau.h"

using namespace nau::scene;

SceneObject* 
SceneObjectFactory::Create (std::string type)
{
	SceneObject *s;
	if (0 == type.compare ("SimpleObject")) {
		s = new SceneObject;
	} 

	else if (0 == type.compare ("OctreeNode")) {
		s =  new OctreeNode;
	}

	else if (0 == type.compare ("Geometry")) {
		s =  new GeometricObject;
	}
	else {
		assert("SceneObjectFactory: type is not valid");
		return 0;
	}

	RENDERMANAGER->addSceneObject(s);
	return s;
}
