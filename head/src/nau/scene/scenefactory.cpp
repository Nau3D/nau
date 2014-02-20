#include <nau/scene/scenefactory.h>
#include <nau/scene/octreescene.h>
#include <nau/scene/octreeByMatscene.h>
#include <nau/scene/scene.h>
#include <nau/scene/sceneposes.h>
#include <nau/scene/sceneskeleton.h>
#include <nau/scene/sceneaux.h>


using namespace nau::scene;




IScene*
SceneFactory::create(std::string scene)
{
	IScene *pScene = 0;

	if ("OctreeByMat" == scene)
		pScene = new OctreeByMatScene;

	if ("Octree" == scene) {
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
