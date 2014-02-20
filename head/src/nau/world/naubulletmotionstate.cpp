#include <nau/world/naubulletmotionstate.h>
#include <nau/math/transformfactory.h>

using namespace nau::world;

NauBulletMotionState::NauBulletMotionState (nau::scene::SceneObject *aObject) : m_SceneObject (aObject)
{
	t = TransformFactory::create("SimpleTransform");
}

NauBulletMotionState::~NauBulletMotionState(void)
{
}

void 
NauBulletMotionState::getWorldTransform (btTransform &worldTrans) const
{
	worldTrans.setFromOpenGLMatrix (m_SceneObject->getTransform().getMat44().getMatrix());
}

// updates scene object transform
// caution, must not access directly the transform
void 
NauBulletMotionState::setWorldTransform (const btTransform &worldTrans)
{
//	worldTrans.getOpenGLMatrix (const_cast<float*> (m_SceneObject->getTransform().getMat44().getMatrix()));
	worldTrans.getOpenGLMatrix (const_cast<float*> (t->getMat44().getMatrix()));
	m_SceneObject->setTransform(t);
}
