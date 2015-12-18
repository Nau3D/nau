#include "nau/world/bulletMotionState.h"
#include "nau/math/matrix.h"

using namespace nau::world;

NauBulletMotionState::NauBulletMotionState (nau::scene::SceneObject *aObject) : m_SceneObject (aObject)
{
}

NauBulletMotionState::~NauBulletMotionState(void)
{
}

void 
NauBulletMotionState::getWorldTransform (btTransform &worldTrans) const
{
	worldTrans.setFromOpenGLMatrix (m_SceneObject->getTransform().getMatrix());
}

// updates scene object transform
// caution, must not access directly the transform
void 
NauBulletMotionState::setWorldTransform (const btTransform &worldTrans)
{
//	worldTrans.getOpenGLMatrix (const_cast<float*> (m_SceneObject->getTransform().getMat44().getMatrix()));
	worldTrans.getOpenGLMatrix (const_cast<float*> (m_Matrix.getMatrix()));
	m_SceneObject->setTransform(m_Matrix);
}
