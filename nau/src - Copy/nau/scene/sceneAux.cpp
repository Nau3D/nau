/*

These scenes are compiled to VAOs at runtime per frame
At the present moment they are being used to draw the camera's frustum

*/

#include "nau/scene/sceneAux.h"
#include "nau/render/renderManager.h"


#include "nau/debug/profile.h"

using namespace nau::scene;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;


SceneAux::SceneAux(void) : Scene() {

	m_Type = "SceneAux";
}


SceneAux::~SceneAux(void) {

}


void
SceneAux::compile() {

}


void SceneAux::unitize() {

}

