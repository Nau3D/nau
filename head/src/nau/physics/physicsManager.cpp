#include "nau/physics/physicsManager.h"

#include "nau.h"
#include "nau/enums.h"
#include "nau/math/data.h"
#include "nau/physics/iPhysics.h"
#include "nau/physics/physicsDummy.h"

using namespace nau::physics;

bool
PhysicsManager::Init() {

	Attribs.add(Attribute(GRAVITY, "GRAVITY", Enums::DataType::VEC4, false, new vec4(0.0f, -9.8f, 0.0f, 0.0f)));

	Attribs.add(Attribute(SCENE_TYPE, "SCENE_TYPE", Enums::DataType::ENUM, false, new NauInt(IPhysics::STATIC)));
	Attribs.listAdd("SCENE_TYPE", "STATIC", IPhysics::STATIC);
	Attribs.listAdd("SCENE_TYPE", "DYNAMIC", IPhysics::RIGID);
	Attribs.listAdd("SCENE_TYPE", "CLOTH", IPhysics::CLOTH);
	Attribs.listAdd("SCENE_TYPE", "PARTICLES", IPhysics::PARTICLES);

	NAU->registerAttributes("PHYSICS_MANAGER", &Attribs);

	return true;
}


AttribSet PhysicsManager::Attribs;
bool PhysicsManager::Inited = Init();
PhysicsManager *PhysicsManager::PhysManInst = NULL;


PhysicsManager*
PhysicsManager::GetInstance() {

	if (!PhysManInst)
		PhysManInst = new PhysicsManager();

	return PhysManInst;
}


PhysicsManager::PhysicsManager() : m_PhysInst(NULL), m_Built(false) {

	m_PhysInst = new PhysicsDummy();
}


PhysicsManager::~PhysicsManager() {

	if (m_PhysInst) {
		delete m_PhysInst;
		m_PhysInst = NULL;
	}

	clear();
}


void 
PhysicsManager::update() {

	if (!m_PhysInst)
		return;

	m_PhysInst->update();

	float *t;
	std::vector<std::shared_ptr<SceneObject>> so;


	for (auto s : m_Scenes) {
		
		switch (s.second) {
		
		case IPhysics::STATIC: break;

		case IPhysics::RIGID:
			t = m_PhysInst->getSceneTransform(s.first->getName());
			s.first->setTransform(math::mat4(t));
			break;

		case IPhysics::CLOTH: 
			s.first->getAllObjects(&so);
			for (auto &o : so) {
				o->getRenderable()->getVertexData()->resetCompilationFlag();
				o->getRenderable()->getVertexData()->compile();
			}
			break;

		case IPhysics::PARTICLES: break;
		}
	}
}


void
PhysicsManager::build() {

	if (!m_Built && m_PhysInst) {
		m_PhysInst->build();
		m_Built = true;
	}
}


void
PhysicsManager::clear() {

	m_MatLib.clear();
	m_Scenes.clear();
	m_Built = false;
}


void
PhysicsManager::addScene(IPhysics::SceneType st, nau::scene::IScene *aScene) {

	m_Scenes[aScene] = st;
	std::string sn = aScene->getName();
	m_PhysInst->setSceneType(sn, st);

	m_PhysInst->setSceneTransform(sn, (float *)aScene->getTransform().getMatrix());

	std::shared_ptr<IRenderable> &r = aScene->getSceneObject(0)->getRenderable();
	std::vector<VertexAttrib> *vd = r->getVertexData()->getDataOf(0).get();
	m_PhysInst->setSceneVertices(sn, (float *)&(vd->at(0)));
	m_PhysInst->setSceneIndices(sn, (unsigned int *)&(r->getIndexData()->getIndexData()->at(0)));

	m_Built = false;
}


PhysicsMaterial &
PhysicsManager::getMaterial(const std::string &name) {

	if (!m_MatLib.count(name))
		m_MatLib[name] = PhysicsMaterial();

	return m_MatLib[name];
}