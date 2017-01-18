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
	Attribs.add(Attribute(TIME_STEP, "TIME_STEP", Enums::DataType::FLOAT, false, new NauFloat(0.016666666667f)));
	Attribs.add(Attribute(CAMERA_POSITION, "CAMERA_POSITION", Enums::DataType::VEC4, true, new vec4(0.0f, 0.0f, -5.0f, 1.0f)));
	Attribs.add(Attribute(CAMERA_DIRECTION, "CAMERA_DIRECTION", Enums::DataType::VEC4, true, new vec4(0.0f, 0.0f, 0.0f, 1.0f)));
	Attribs.add(Attribute(CAMERA_UP, "CAMERA_UP", Enums::DataType::VEC4, false, new vec4(0.0f, 1.0f, 0.0f, 1.0f)));
	Attribs.add(Attribute(CAMERA_RADIUS, "CAMERA_RADIUS", Enums::DataType::FLOAT, true, new NauFloat(1.0f)));
	Attribs.add(Attribute(CAMERA_HEIGHT, "CAMERA_HEIGHT", Enums::DataType::FLOAT, true, new NauFloat(1.0f)));

#ifndef _WINDLL
	NAU->registerAttributes("PHYSICS_MANAGER", &Attribs);
#endif

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


PhysicsManager::PhysicsManager() : m_PhysInst(NULL), m_Built(false), hasCamera(false) {

	registerAndInitArrays(Attribs);

	m_PhysInst = loadPlugin();

	m_PropertyManager = PhysicsPropertyManager::GetInstance();
	if (!m_PhysInst)
		return;

	m_PhysInst->setPropertyManager(m_PropertyManager);

	std::map < std::string, IPhysics::Prop> &props = m_PhysInst->getGlobalProperties();

	int k = 0;
	int floatCount = Attribs.getDataTypeCount(Enums::FLOAT);
	int vecCount = Attribs.getDataTypeCount(Enums::VEC4);
	for (auto &p : props) {
		Enums::DataType dt = p.second.propType == IPhysics::FLOAT ? Enums::FLOAT : Enums::VEC4;
		if (p.second.propType == IPhysics::FLOAT) 
			Attribs.add(Attribute(++floatCount, p.first, Enums::FLOAT, false, new NauFloat(p.second.x)));
		else
			Attribs.add(Attribute(++vecCount, p.first, Enums::VEC4, false, new vec4(p.second.x, p.second.y, p.second.z, p.second.w)));
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
		delete m_PropertyManager;
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
			SLOG("%s: Invalid Plugin DLL:  'init', 'createPhys' and 'getClassName' must be defined", fn.c_str());
			return NULL;
		}
		else
			loaded++;

		initFunc();
		
		// push the objects and modules into our vectors
		char *s = getClassNameFunc();
		SLOG("Physics plugin %s (%s) loaded successfully", fn.c_str(), s);
	

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
			t = m_PhysInst->getSceneTransform(s.first->getName());
			s.first->setTransform(math::mat4(t));
			s.first->getAllObjects(&so);
			for (auto &o : so) {
				o->getRenderable()->getVertexData()->resetCompilationFlag();
				o->getRenderable()->getVertexData()->compile();
			}
			break;

		case IPhysics::CHARACTER:
			t = m_PhysInst->getSceneTransform(s.first->getName());
			s.first->setTransform(math::mat4(t));
			break;

		case IPhysics::PARTICLES:
		{
			std::string &sceneName = s.first->getName();
			PhysicsMaterial &pm = getMaterial(s.second);
			int nPart = static_cast<int>(pm.getPropf((FloatProperty)pm.getAttribSet()->getAttributes()["NBPARTICLES"]->getId()));
			std::string bufferName = pm.getProps((StringProperty)pm.getAttribSet()->getAttributes()["BUFFER"]->getId());
			IBuffer * pointsBuffer = RESOURCEMANAGER->getBuffer(bufferName);
			pointsBuffer->setSubData(0, nPart * 4 * sizeof(float), pm.getBuffer());
			RENDERMANAGER->getCurrentPass()->setPropui(Pass::INSTANCE_COUNT, nPart);
		}
			break;
		case IPhysics::DEBUG:
		{
			IBuffer * b = RESOURCEMANAGER->getBufferByID(s.first->getSceneObject(0)->getRenderable()->getVertexData()->getBufferID(VertexData::GetAttribIndex(std::string("position"))));
			std::vector<float> * debugPos = m_PhysInst->getDebug();
			b->setData(debugPos->size() * sizeof(float), &debugPos->at(0));
			s.first->getAllObjects(&so);
			for (auto &o : so) {
				o->getRenderable()->getVertexData()->resetCompilationFlag();
				o->getRenderable()->getVertexData()->compile();
			}
		}
			break;
		}
	}

	for (auto cam : *(m_PhysInst->getCameraPositions())) {
		Camera * camera = RENDERMANAGER->getCamera(cam.first).get();
		vec4 previous = camera->getPropf4(Camera::POSITION);
		vec4 current = vec4(cam.second[0], cam.second[1], cam.second[2], 1.0f);
		if (current != previous)
			camera->setPropf4(Camera::POSITION, current);
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
	initArrays();
}


void
PhysicsManager::addScene(nau::scene::IScene *aScene, const std::string &matName) {

	if (!m_PhysInst)
		return;

	m_Scenes[aScene] = matName;
	std::string sn = aScene->getName();
	PhysicsMaterial &pm = getMaterial(matName);
	IPhysics::SceneType type = (IPhysics::SceneType)pm.getPrope(PhysicsMaterial::SCENE_TYPE);
	IPhysics::SceneShape shape = (IPhysics::SceneShape)pm.getPrope(PhysicsMaterial::SCENE_SHAPE);

	m_PhysInst->setSceneType(sn, type);

	float * max = new float[3]();
	float * min = new float[3]();
	vec3 maxVec = aScene->getBoundingVolume().getMax();
	vec3 minVec = aScene->getBoundingVolume().getMin();
	max[0] = maxVec.x; max[1] = maxVec.y; max[2] = maxVec.z;
	min[0] = minVec.x; min[1] = minVec.y; min[2] = minVec.z;

	m_PhysInst->setSceneShape(sn, shape, min, max);

	m_PhysInst->setSceneCondition(sn, (IPhysics::SceneCondition)pm.getPrope(PhysicsMaterial::SCENE_CONDITION));
	
	switch (type) {
	case IPhysics::PARTICLES: 
	{
		int maxParticles = static_cast<int>(pm.getPropf((FloatProperty)pm.getAttribSet()->getAttributes()["MAX_PARTICLES"]->getId()));
		pm.setBuffer((float *)malloc(maxParticles * 4 * sizeof(float)));
		IBuffer * buff = RESOURCEMANAGER->getBuffer(pm.getProps((StringProperty)pm.getAttribSet()->getAttributes()["BUFFER"]->getId()));
		buff->setData(maxParticles * 4 * sizeof(float), pm.getBuffer());
		m_PhysInst->setScene(
			sn,
			matName,
			maxParticles,
			pm.getBuffer(),
			0,
			NULL,
			(float *)aScene->getTransform().getMatrix()
		);
	}
		break;

	case IPhysics::DEBUG:
		m_PhysInst->setScene(sn, matName, 0, NULL, 0, NULL, (float *)aScene->getTransform().getMatrix());
		break;

	default:
		m_PhysInst->setSceneTransform(sn, (float *)aScene->getTransform().getMatrix());
		std::shared_ptr<IRenderable> &r = aScene->getSceneObject(0)->getRenderable();
		std::vector<VertexAttrib> *vd = r->getVertexData()->getDataOf(0).get();
		m_PhysInst->setScene(
			sn,
			matName,
			static_cast<int> (vd->size()),
			(float *)&(vd->at(0)),
			static_cast<int> (r->getIndexData()->getIndexData()->size()),
			(unsigned int *)&(r->getIndexData()->getIndexData()->at(0)),
			(float *)aScene->getTransform().getMatrix()
		);
		EVENTMANAGER->addListener("SCENE_TRANSFORM", this);
		break;
	}

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

void nau::physics::PhysicsManager::cameraAction(Camera * camera, std::string action, float * value) {
	if (!m_PhysInst)
		return;
	m_PhysInst->setCameraAction(camera->getName(), action, value);
}


bool 
PhysicsManager::isPhysicsAvailable() {

	return (NULL != m_PhysInst);
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
PhysicsManager::eventReceived(const std::string & sender, const std::string & eventType, const std::shared_ptr<IEventData>& evt) {

	if (m_PhysInst && eventType == "SCENE_TRANSFORM") {
		std::string * strEvt = (std::string*) evt->getData();
		IScene * scene = RENDERMANAGER->getScene(*strEvt).get();
		m_PhysInst->setSceneTransform(scene->getName(), (float *)scene->getTransform().getMatrix());
	}
}


std::string & nau::physics::PhysicsManager::getName() {

	return *(new std::string("PHYSICS_MANAGER"));
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

	int id = PhysicsMaterial::Attribs.getID(property);

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