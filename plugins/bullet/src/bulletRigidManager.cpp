#include "bulletRigidManager.h"



BulletRigidManager::BulletRigidManager() {
}


BulletRigidManager::~BulletRigidManager() {
	delete &rigidBodies;
}

void BulletRigidManager::update() {
}

void BulletRigidManager::createInfo(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform) {
	rigidBodies[scene].extInfo = externalInfo(nbVertices, vertices, nbIndices, indices, transform);
}

btCollisionShape * BulletRigidManager::createCollisionShape(const std::string & scene, nau::physics::IPhysics::BoundingVolume shape, bool isStatic) {
	switch (shape.sceneShape)
	{
	case nau::physics::IPhysics::BOX:
		return new btBoxShape(btVector3(shape.max[0], shape.max[1], shape.max[2]));
	case nau::physics::IPhysics::SPHERE:
		return new btSphereShape(shape.max[0]);
	case nau::physics::IPhysics::CAPSULE:
		return new btCapsuleShape(shape.max[0], shape.max[1]);
	default:
	{
		btTriangleIndexVertexArray * indexVertexArrays = new btTriangleIndexVertexArray();

		btIndexedMesh * mesh = new btIndexedMesh();
		mesh->m_numTriangles = rigidBodies[scene].extInfo.nbIndices / 3;
		mesh->m_triangleIndexBase = reinterpret_cast<const unsigned char *>(rigidBodies[scene].extInfo.indices);
		mesh->m_triangleIndexStride = 3 * sizeof(unsigned int);
		mesh->m_numVertices = rigidBodies[scene].extInfo.nbVertices;
		mesh->m_vertexBase = reinterpret_cast<const unsigned char *>(rigidBodies[scene].extInfo.vertices);
		mesh->m_vertexStride = 4 * sizeof(float);

		indexVertexArrays->addIndexedMesh(*mesh, PHY_INTEGER);

		if (isStatic) {
			bool useQuantizedAabbCompression = true;
			return new btBvhTriangleMeshShape(indexVertexArrays, useQuantizedAabbCompression);
		}
		else {
			btGImpactMeshShape *gImpa = new btGImpactMeshShape(indexVertexArrays);
			gImpa->updateBound();
			return gImpa;
		}
	}
	}
}

btRigidBody * BulletRigidManager::addRigid(const std::string & scene, btCollisionShape * shape, float mass, bool isStatic) {
	btVector3 localInertia(0, 0, 0);
	BulletMotionState * motionState = new BulletMotionState(rigidBodies[scene].extInfo.transform);
	btRigidBody * body;
	if (isStatic) {
		if (scene.compare("plane") == 0) {
			btRigidBody::btRigidBodyConstructionInfo rbGroundInfo(0, motionState, new btStaticPlaneShape(btVector3(0, 1, 0), 0));
			body = new btRigidBody(rbGroundInfo);
		}
		else {
			body = new btRigidBody(0, motionState, shape, localInertia);
		}
		body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_STATIC_OBJECT);
	}
	else {
		shape->calculateLocalInertia(mass, localInertia);
		body = new btRigidBody(mass, motionState, shape, localInertia);
	}
	body->setUserPointer(static_cast<void *>(new std::string(scene)));
	rigidBodies[scene].object = body;
	return body;
}

void BulletRigidManager::setMass(std::string name, float value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		btRigidBody * body = btRigidBody::upcast(rigidBodies[name].object);
		if (body) {
			btVector3 localInertia = body->getLocalInertia();
			body->setMassProps(value, localInertia);
		}
	}
}

void BulletRigidManager::setLocalInertia(std::string name, float * value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		btRigidBody * body = btRigidBody::upcast(rigidBodies[name].object);
		if (body) {
			float mass = 1.0f / body->getInvMass();
			body->setMassProps(mass, btVector3(value[0], value[1], value[2]));
			body->updateInertiaTensor();
		}
	}
}

void BulletRigidManager::setFriction(std::string name, float value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		btRigidBody * body = btRigidBody::upcast(rigidBodies[name].object);
		if (body) {
			body->setFriction(value);
		}
	}
}

void BulletRigidManager::setRollingFriction(std::string name, float value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		btRigidBody * body = btRigidBody::upcast(rigidBodies[name].object);
		if (body) {
			body->setRollingFriction(value);
		}
	}
}

void BulletRigidManager::setRestitution(std::string name, float value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		btRigidBody * body = btRigidBody::upcast(rigidBodies[name].object);
		if (body) {
			body->setRestitution(value);
		}
	}
}

void BulletRigidManager::addImpulse(std::string name, float * value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		btRigidBody * body = btRigidBody::upcast(rigidBodies[name].object);
		if (body) {
			if (!body->isActive())
				body->activate();
			btVector3 impulse = btVector3(value[0], value[1], value[2]);
			body->applyCentralImpulse(impulse);
		}
	}
}

void BulletRigidManager::move(std::string scene, float * transform) {
	if (rigidBodies.find(scene) != rigidBodies.end()) {
		btRigidBody * body = btRigidBody::upcast(rigidBodies[scene].object);
		if (body) {
			rigidBodies[scene].extInfo.transform = transform;
			btTransform trans;
			trans.setFromOpenGLMatrix(transform);
			body->getMotionState()->getWorldTransform(trans);
		}
	}
}


