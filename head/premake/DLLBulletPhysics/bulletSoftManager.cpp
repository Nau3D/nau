#include "bulletSoftManager.h"

BulletSoftManager::BulletSoftManager() {
}


BulletSoftManager::~BulletSoftManager() {
	delete &softBodies;
}

void BulletSoftManager::update() {
	for (auto scene : softBodies) {
		btSoftBody * cloth = btSoftBody::upcast(scene.second.info.object);

		float m[16];
		cloth->getWorldTransform().getOpenGLMatrix(m);
		for (int k = 0; k < 16; k++) { scene.second.info.extInfo.transform[k] = m[k]; }

		btSoftBody::tNodeArray& nodes(cloth->m_nodes);
		for (int i = 0; i < nodes.size(); i++) {
			for (int j = 0; j < 3; j++)
				scene.second.info.extInfo.vertices[(4 * i) + j] = nodes[i].m_x.m_floats[j];
		}
		btSoftBody::PSolve_Anchors(cloth, 0.0f, 0.0f);
	}
}

void BulletSoftManager::createInfo(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform, int condition, float * conditionValue) {
	softBodies[scene].info.extInfo= externalInfo(nbVertices, vertices, nbIndices, indices, transform);
	softBodies[scene].condition = condition;
	softBodies[scene].contidionPlane = new btStaticPlaneShape(btVector3(conditionValue[0], conditionValue[1], conditionValue[2]), conditionValue[3]);
}

btSoftBody * BulletSoftManager::addSoftBody(btSoftBodyWorldInfo & worldInfo, const std::string & scene) {
	float * newPoints = new float[(softBodies[scene].info.extInfo.nbVertices* 3)];
	std::vector<int> lockedIndices;
	float * auxPoint = new float[3];
	for (int i = 0; i < softBodies[scene].info.extInfo.nbVertices; i++) {
		for (int j = 0; j < 3; j++) {
			newPoints[(i * 3) + j] = softBodies[scene].info.extInfo.vertices[(i * 4) + j];
			auxPoint[j] = newPoints[(i * 3) + j];
		}
		if (isLocked(scene, btVector3(auxPoint[0], auxPoint[1], auxPoint[2])))
			lockedIndices.push_back(i);
	}

	btSoftBody * cloth = btSoftBodyHelpers::CreateFromTriMesh(
		worldInfo,
		newPoints,
		reinterpret_cast<const int *>(softBodies[scene].info.extInfo.indices),
		static_cast<int>(softBodies[scene].info.extInfo.nbIndices / 3)
	);
	/*btSoftBody::Material * pm = cloth->appendMaterial();
	pm->m_kLST = 0.5;
	cloth->generateBendingConstraints(2, pm);
	cloth->m_cfg.viterations = 50;
	cloth->m_cfg.piterations = 50;
	cloth->m_cfg.kDF = 0.5;
	cloth->randomizeConstraints();*/

	//cloth->setTotalMass(0.2f);
	//cloth->setWindVelocity(btVector3(9.0f, 0.0, -1.0f));
	//cloth->addForce(btVector3(1.0f, 0.0f, -1.0f));
	//cloth->m_cfg.aeromodel = btSoftBody::eAeroModel::V_TwoSidedLiftDrag;

	float m[16];
	for (int i = 0; i < 16; i++) { m[i] = softBodies[scene].info.extInfo.transform[i]; }
	btTransform * trans = new btTransform();
	trans->setFromOpenGLMatrix(m);
	//cloth->setWorldTransform(*trans);
	cloth->transform(*trans);

	if (lockedIndices.size() > 0) {
		btTransform * startTransform = new btTransform();
		startTransform->setIdentity();
		startTransform->setOrigin(btVector3(0.0f, 0.0f, 0.0f));
		btDefaultMotionState * motion = new btDefaultMotionState(*startTransform);
		btRigidBody * body = new btRigidBody(0, motion, new btBoxShape(btVector3(1.0f, 1.0f, 1.0f)));
		for(int indice : lockedIndices)
			cloth->appendAnchor(indice, body);
	}

	cloth->setUserPointer(static_cast<void *>(new std::string(scene)));

	softBodies[scene].info.object = cloth;
	return cloth;
}

void BulletSoftManager::setViterations(const std::string & scene, float value) {
	if (value > 0.0f) {
		btSoftBody * cloth = getCloth(scene);
		if (cloth)
			cloth->m_cfg.viterations = (int)value;
	}
}

void BulletSoftManager::setPiteration(const std::string & scene, float value) {
	if (value > 0.0f) {
		btSoftBody * cloth = getCloth(scene);
		if (cloth)
			cloth->m_cfg.piterations = (int)value;
	}
}

void BulletSoftManager::move(std::string scene, float * transform) {
	btSoftBody * cloth = getCloth(scene);
	if (cloth) {
		float m[16];
		for (int i = 0; i < 16; i++) { m[i] = transform[i]; }
		btTransform * trans = new btTransform();
		trans->setFromOpenGLMatrix(m);
		cloth->transform(*trans);
		//cloth->setWorldTransform(*trans);
	}
}

bool BulletSoftManager::isLocked(const std::string & scene, btVector3 vertex) {
	switch (softBodies[scene].condition)
	{
	case CLOTH_CONDITION_GT:
		return distance(scene, vertex) > 0.0f;
	case CLOTH_CONDITION_LT:
		return distance(scene, vertex) < 0.0f;
	case CLOTH_CONDITION_EGT:
		return contains(scene, vertex) || distance(scene, vertex) > 0.0f;
	case CLOTH_CONDITION_ELT:
		return contains(scene, vertex) || distance(scene, vertex) < 0.0f;
	case CLOTH_CONDITION_EQ:
		return contains(scene, vertex);
	default:
		return false;
	}
}

float BulletSoftManager::distance(const std::string & scene, btVector3 p) {
	return p.dot(softBodies[scene].contidionPlane->getPlaneNormal()) + softBodies[scene].contidionPlane->getPlaneConstant();
}

bool BulletSoftManager::contains(const std::string & scene, btVector3 p) {
	return abs(distance(scene, p)) < (1.0e-7f);
}

btSoftBody * BulletSoftManager::getCloth(const std::string & scene) {
	if (softBodies.find(scene) != softBodies.end())
		return btSoftBody::upcast(softBodies[scene].info.object);
	return NULL;
}
