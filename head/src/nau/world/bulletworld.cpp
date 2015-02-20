#include "nau/world/bulletworld.h"

#include "nau/geometry/iboundingvolume.h"
#include "nau/world/naubulletmotionstate.h"
#include "nau/render/vertexdata.h"
#include "nau/material/materialgroup.h"

using namespace nau::world;
using namespace nau::geometry;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::material;


BulletWorld::BulletWorld(void): m_pScene (0), m_pDynamicsWorld (0)
{
}


BulletWorld::~BulletWorld(void)
{
	delete m_pDynamicsWorld;
}


void 
BulletWorld::update (void)
{
	if (0 != m_pDynamicsWorld) {
		m_pDynamicsWorld->stepSimulation(1 / 120.0f);
		//m_pDynamicsWorld->updateAabbs();
	}
}


void 
BulletWorld::build (void) /***MARK***/ //I'm assuming all objects inside scene are static objects
{
	if (0 != m_pScene) {
		if (0 != m_pDynamicsWorld) {
			delete m_pDynamicsWorld;
			m_pDynamicsWorld = 0;
		}

		btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
		btCollisionDispatcher* dispatcher = new	btCollisionDispatcher (collisionConfiguration);

		IBoundingVolume &sceneAabb = m_pScene->getBoundingVolume();

		btVector3 worldAabbMin  (sceneAabb.getMin().x, sceneAabb.getMin().y, sceneAabb.getMin().z);
		btVector3 worldAabbMax  (sceneAabb.getMax().x, sceneAabb.getMax().y, sceneAabb.getMax().z);

		btBroadphaseInterface* broadphase = new btAxisSweep3 (worldAabbMin,worldAabbMax);//  new btMultiSapBroadphase();//new btAxisSweep3(worldAabbMin,worldAabbMax,maxProxies);
		btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

		m_pDynamicsWorld = new btDiscreteDynamicsWorld(dispatcher,broadphase,solver);
		m_pDynamicsWorld->setGravity(btVector3(0,-10,0)); /***MARK***/ //Should be user definable

		std::vector<SceneObject*> &sceneObjects = m_pScene->getAllObjects();

		std::vector<SceneObject*>::iterator sceneObjectsIter;

		sceneObjectsIter = sceneObjects.begin();

		for (; sceneObjectsIter != sceneObjects.end(); sceneObjectsIter++) {
			VertexData &vd = (*sceneObjectsIter)->getRenderable().getVertexData();
			

			std::vector<MaterialGroup*> &matGroups = (*sceneObjectsIter)->getRenderable().getMaterialGroups();
			std::vector<MaterialGroup*>::iterator matGroupsIter;

			matGroupsIter = matGroups.begin();

			for ( ; matGroupsIter != matGroups.end(); matGroupsIter++) {

				if ((*matGroupsIter)->getIndexData().getIndexSize()) {
				
					std::vector<unsigned int> &indexes = (*matGroupsIter)->getIndexData().getIndexData();
					btTriangleIndexVertexArray* indexVertexArrays = new btTriangleIndexVertexArray (
							static_cast<int> (indexes.size() / 3), 
							reinterpret_cast<int *>(&indexes[0]),
							3 * sizeof(unsigned int),
							static_cast<int> (vd.getDataOf(VertexData::getAttribIndex("position")).size()),
							reinterpret_cast<btScalar*>(&(vd.getDataOf(VertexData::getAttribIndex("position"))[0])),
							/*3*/4 * sizeof (float)
						);


					bool useQuantizedAabbCompression = true;
					btBvhTriangleMeshShape *trimeshShape  = new btBvhTriangleMeshShape(indexVertexArrays,useQuantizedAabbCompression);

					NauBulletMotionState *motionState = new NauBulletMotionState ((*sceneObjectsIter));

					btVector3 localInertia (0, 0, 0);
					btRigidBody* body = new btRigidBody(0,motionState,trimeshShape,localInertia);

					if (0 != (*sceneObjectsIter)->getName().compare ("pPlane1")){
						body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_STATIC_OBJECT);
					} else {
						m_RigidBodies["water"] = body;

						body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
						body->setActivationState (DISABLE_DEACTIVATION);
					}
					body->setRestitution (0.0f);

					m_pDynamicsWorld->addRigidBody (body);
				}
			}
		}
	}
}


void 
BulletWorld::setScene (nau::scene::IScene *aScene) 
{
	m_pScene = aScene;
}

void 
BulletWorld::_add (float mass, nau::scene::SceneObject *aObject, std::string name, nau::math::vec3 aVec)
{
	//btCollisionShape *aShape = new btBoxShape (btVector3 (aVec.x, aVec.y, aVec.z));
	btCollisionShape *aShape = new btSphereShape(aVec.z);
	NauBulletMotionState *motionState = new NauBulletMotionState (aObject);

	btVector3 localInertia (0,0,0);
	aShape->calculateLocalInertia (mass, localInertia);
	btRigidBody* body = new btRigidBody (mass, motionState, aShape, localInertia);
	body->setFriction(2.0);

	m_RigidBodies[name] = body;
	m_RigidBodies[name]->setActivationState (DISABLE_DEACTIVATION);
	body->setAngularFactor (0.0f);
	body->setRestitution (0.0f);

	m_pDynamicsWorld->addRigidBody (body);
}

void 
BulletWorld::setKinematic (std::string name)
{
	m_RigidBodies[name]->setCollisionFlags (m_RigidBodies[name]->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
	m_RigidBodies[name]->setActivationState (DISABLE_DEACTIVATION);
}

void 
BulletWorld::setDynamic (std::string name)
{
	m_RigidBodies[name]->setCollisionFlags (m_RigidBodies[name]->getCollisionFlags() ^ btCollisionObject::CF_KINEMATIC_OBJECT);
	m_RigidBodies[name]->activate();
}

void 
BulletWorld::setVelocity (std::string name, nau::math::vec3 vel)
{
	if (m_RigidBodies.count(name)) {
		//m_RigidBodies[name]->setLinearVelocity (btVector3 (vel.x, vel.y, vel.z));
		if (m_RigidBodies[name]->getVelocityInLocalPoint (m_RigidBodies[name]->getCenterOfMassPosition()).length() < 10.0f) {
			m_RigidBodies[name]->applyCentralImpulse (btVector3 (3*vel.x, 3*vel.y, 3*vel.z));//, m_RigidBodies[name]->getCenterOfMassPosition()); /**???**/
		}
	}
}

void 
BulletWorld::disableObject (std::string name)
{
	m_pDynamicsWorld->removeCollisionObject (m_RigidBodies[name]);
}

void 
BulletWorld::enableObject (std::string name)
{
	m_pDynamicsWorld->addCollisionObject (m_RigidBodies[name]);
}
