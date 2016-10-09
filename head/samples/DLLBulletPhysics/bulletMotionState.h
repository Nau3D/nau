#ifndef _BULLETMOTIONSTATE_H
#define _BULLETMOTIONSTATE_H

#include <btBulletDynamicsCommon.h>

class BulletMotionState : public btMotionState {

private:
	float * transform;

public:
	BulletMotionState(float * trans);
	~BulletMotionState();

	/*btMotionState interface*/
	void getWorldTransform(btTransform &worldTrans) const;
	void setWorldTransform(const btTransform &worldTrans);
};

#endif