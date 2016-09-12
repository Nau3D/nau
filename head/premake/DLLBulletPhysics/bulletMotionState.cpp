#include "bulletMotionState.h"



BulletMotionState::BulletMotionState(float * trans) {
	transform = trans;
}


BulletMotionState::~BulletMotionState() {
}

void BulletMotionState::getWorldTransform(btTransform & worldTrans) const {
	worldTrans.setFromOpenGLMatrix(transform);
}

void BulletMotionState::setWorldTransform(const btTransform & worldTrans) {
	worldTrans.getOpenGLMatrix(transform);

}
