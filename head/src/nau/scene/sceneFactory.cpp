#include "nau/scene/sceneFactory.h"
#include "nau/scene/octreeScene.h"
#include "nau/scene/octreeUnified.h"
#include "nau/scene/octreeByMatScene.h"
#include "nau/scene/scene.h"
#include "nau/scene/scenePoses.h"
#include "nau/scene/sceneSkeleton.h"
#include "nau/scene/sceneAux.h"


using namespace nau::scene;




IScene*
SceneFactory::Create(std::string scene)
{
	IScene *pScene = 0;

	if ("OctreeUnified" == scene) {
		pScene = new OctreeUnified;
	}
	else if ("OctreeByMat" == scene) {
		pScene = new OctreeByMatScene;
	}
	else if ("Octree" == scene) {
		pScene = new OctreeScene;
	}
	else if ("Scene" == scene) {
		pScene = new Scene;
	}
	else if ("ScenePoses" == scene) {
		pScene = new ScenePoses;
	}
	else if ("SceneSkeleton" == scene) {
		pScene = new SceneSkeleton;
	}
	else if ("SceneAux" == scene) {
		pScene = new SceneAux;
	}

	return pScene;
}
