#include "nau/physics/physicsManager.h"

#include "nau.h"
#include "nau/enums.h"
#include "nau/slogger.h"
#include "nau/math/data.h"
#include "nau/physics/iPhysics.h"
#include "nau/physics/physicsPropertyManager.h"
#include "nau/physics/physicsDummy.h"
#include "nau/system/file.h"

#include <stdexcept>
#include <exception>
#include <windows.h>

using namespace nau::physics;

typedef int(*deletePhysicsProc)();
deletePhysicsProc deletePhysics;


bool
PhysicsManager::Init() {

	//Attribs.add(Attribute(GRAVITY, "GRAVITY", Enums::DataType::VEC4, false, new vec4(0.0f, -9.8f, 0.0f, 0.0f)));


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

	registerAndInitArrays(Attribs);
	m_PhysInst = loadPlugin();
	m_PropertyManager = PhysicsPropertyManager::GetInstance();
	if (!m_PhysInst)
		return;

	m_PhysInst->setPropertyManager(m_PropertyManager);

	std::map < std::string, IPhysics::Prop> &props = m_PhysInst->getGlobalProperties();

	int k = 0;
	for (auto &p : props) {
		Enums::DataType dt = p.second.propType == IPhysics::FLOAT ? Enums::FLOAT : Enums::VEC4;
		if (p.second.propType == IPhysics::FLOAT) 
			Attribs.add(Attribute(k++, p.first, Enums::FLOAT, false, new NauFloat(p.second.x)));
		else
			Attribs.add(Attribute(k++, p.first, Enums::VEC4, false, new vec4(p.second.x, p.second.y, p.second.z, p.second.w)));
	}

	
	std::map<std::string, nau::physics::IPhysics::Prop> &propsM = m_PhysInst->getMaterialProperties();
	k = 0;
	for (auto &p : propsM) {
		Enums::DataType dt = p.second.propType == IPhysics::FLOAT ? Enums::FLOAT : Enums::VEC4;
		if (p.second.propType == IPhysics::FLOAT)
			PhysicsMaterial::Attribs.add(Attribute(k++, p.first, Enums::FLOAT, false, new NauFloat(p.second.x)));
		else
			PhysicsMaterial::Attribs.add(Attribute(k++, p.first, Enums::VEC4, false, new vec4(p.second.x, p.second.y, p.second.z, p.second.w)));
	}
}


PhysicsManager::~PhysicsManager() {

	if (m_PhysInst) {
		deletePhysics();
		m_PhysInst = NULL;
	}

	clear();
}


IPhysics *
PhysicsManager::loadPlugin() {

	std::vector<std::string> files;
#ifdef _DEBUG
	nau::system::File::GetFilesInFolder(".\\nauSettings\\plugins_d\\physics\\", "dll", &files);
#else
	nau::system::File::GetFilesInFolder(".\\nauSettings\\plugins\\physics\\", "dll", &files);
#endif
	typedef void (__cdecl *initProc)(void);
	typedef void *(__cdecl *createPhysics)(void);
	typedef char *(__cdecl *getClassNameProc)(void);
	int loaded = 0;

	if (files.size() == 0)
		return NULL;

	std::string fn = files[0];

		wchar_t wtext[256];
		mbstowcs(wtext, fn.c_str(), fn.size() + 1);//Plus null
		LPWSTR ptr = wtext;
		HINSTANCE mod = LoadLibraryA(fn.c_str());

		if (!mod) {
			//SLOG("Library %s wasn't loaded successfully!", fn.c_str());
			return NULL;
		}

		initProc initFunc = (initProc)GetProcAddress(mod, "init");
		createPhysics createPhys = (createPhysics)GetProcAddress(mod, "createPhysics");
		getClassNameProc getClassNameFunc = (getClassNameProc)GetProcAddress(mod, "getClassName");
		deletePhysics = (deletePhysicsProc)GetProcAddress(mod, "deletePhysics");
		if (!initFunc || !createPhys || !getClassNameFunc) {
			//SLOG("%s: Invalid Plugin DLL:  'init', 'createPhys' and 'getClassName' must be defined", fn.c_str());
			return NULL;
		}
		else
			loaded++;

		initFunc();
		
		// push the objects and modules into our vectors
		char *s = getClassNameFunc();
		//SLOG("Physics plugin %s (%s) loaded successfully", fn.c_str(), s);
	

		IPhysics *ip = (IPhysics *)createPhys();

	// Close the file when we are done
	return ip;
}


void 
PhysicsManager::update() {

	if (!m_PhysInst)
		return;

	m_PhysInst->update();

	float *t;
	std::vector<std::shared_ptr<SceneObject>> so;


	for (auto s : m_Scenes) {
		
		int st = getMaterial(s.second).getPrope(PhysicsMaterial::SCENE_TYPE);
		switch (st) {
		
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
		// Falta passar aqui as propriedades globais
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
PhysicsManager::addScene(nau::scene::IScene *aScene, const std::string &matName) {

	if (!m_PhysInst)
		return;

	m_Scenes[aScene] = matName;
	std::string sn = aScene->getName();
	PhysicsMaterial &pm = getMaterial(matName);

	m_PhysInst->setSceneType(sn, (IPhysics::SceneType)pm.getPrope(PhysicsMaterial::SCENE_TYPE));

	m_PhysInst->setSceneTransform(sn, (float *)aScene->getTransform().getMatrix());

	std::shared_ptr<IRenderable> &r = aScene->getSceneObject(0)->getRenderable();
	std::vector<VertexAttrib> *vd = r->getVertexData()->getDataOf(0).get();
	m_PhysInst->setScene(sn, (float *)&(vd->at(0)), 
		(unsigned int *)&(r->getIndexData()->getIndexData()->at(0)), 
		(float *)aScene->getTransform().getMatrix());

	std::map<std::string, std::unique_ptr<Attribute>> &attrs = pm.getAttribSet()->getAttributes();
	
	for (auto &a : attrs) {
		switch (a.second->getType()) {
			case Enums::FLOAT: 
				m_PhysInst->applyFloatProperty(sn, a.second->getName(), pm.getPropf((FloatProperty)a.second->getId())); break;
			case Enums::VEC4: 
				m_PhysInst->applyVec4Property(sn, a.second->getName(), &(pm.getPropf4((Float4Property)a.second->getId()).x)); break;
	
		}
	}

	m_Built = false;
}


void
PhysicsManager::updateProps() {

	if (!m_PhysInst)
		return;

	std::map<std::string, std::unique_ptr<Attribute>> &attrs = getAttribSet()->getAttributes();

	for (auto &a : attrs) {
		switch (a.second->getType()) {
		case Enums::FLOAT:
			m_PhysInst->applyGlobalFloatProperty(a.second->getName(), getPropf((FloatProperty)a.second->getId())); break;
		case Enums::VEC4:
			m_PhysInst->applyGlobalVec4Property(a.second->getName(), &(getPropf4((Float4Property)a.second->getId()).x)); break;

		}
	}
}


void 
PhysicsManager::setPropf(FloatProperty p, float value) {

	m_FloatProps[p] = value;
	applyGlobalFloatProperty(Attribs.getName(p, Enums::FLOAT), value);
}


void 
PhysicsManager::setPropf4(Float4Property p, vec4 &value) {

	m_Float4Props[p] = value;
	applyGlobalVec4Property(Attribs.getName(p, Enums::VEC4), &value.x);
}


void
PhysicsManager::applyGlobalFloatProperty(const std::string &property, float value) {

	if (!m_PhysInst)
		return;
	
	m_PhysInst->applyGlobalFloatProperty(property, value);
}


void
PhysicsManager::applyGlobalVec4Property(const std::string &property, float *value) {

	if (!m_PhysInst)
		return;

	m_PhysInst->applyGlobalVec4Property(property, value);
}


void
PhysicsManager::applyMaterialFloatProperty(const std::string &matName, const std::string &property, float value) {

	if (!m_PhysInst || m_MatLib.count(matName) == 0)
		return;

	int id = PhysicsMaterial::Attribs.getID(property);

	if (id != -1) {
		for (auto &sc : m_Scenes) {
			if (sc.second == matName)
				m_PhysInst->applyFloatProperty(sc.first->getName(), property, value);
		}
		
	}

}


void
PhysicsManager::applyMaterialVec4Property(const std::string &matName, const std::string &property, float *value) {

	if (!m_PhysInst)
		return;

	int id = Attribs.getID(property);

	if (id != -1) {
		for (auto &sc : m_Scenes) {
			if (sc.second == matName)
				m_PhysInst->applyVec4Property(sc.first->getName(), property, value);
		}
	}
}


PhysicsMaterial &
PhysicsManager::getMaterial(const std::string &name) {

	if (!m_MatLib.count(name))
		m_MatLib[name] = PhysicsMaterial(name);

	return m_MatLib[name];
}


void
PhysicsManager::getMaterialNames(std::vector<std::string> *v) {

	for (auto s : m_MatLib) {
		v->push_back(s.first);
	}
}