#include "nau/physics/physicsPropertyManager.h"

using namespace nau;
using namespace nau::physics;

PhysicsPropertyManager *PhysicsPropertyManager::Instance = NULL;

PhysicsPropertyManager*
PhysicsPropertyManager::GetInstance() {

	if (!Instance)
		Instance = new PhysicsPropertyManager();

	return Instance;
}


PhysicsPropertyManager::PhysicsPropertyManager() {

	m_PhysicsManager = NULL;
}


float 
PhysicsPropertyManager::getMaterialFloatProperty(const std::string &material, const std::string &property) {

	if (!m_PhysicsManager)
		m_PhysicsManager = PhysicsManager::GetInstance();

	PhysicsMaterial &m = m_PhysicsManager->getMaterial(material);
	int id = m.getAttribSet()->getID(property); 
	return m.getPropf((AttributeValues::FloatProperty)id);
}


float *
PhysicsPropertyManager::getMaterialVec4Property(const std::string &material, const std::string &property) {

	if (!m_PhysicsManager)
		m_PhysicsManager = PhysicsManager::GetInstance();

	PhysicsMaterial &m = m_PhysicsManager->getMaterial(material);
	int id = m.getAttribSet()->getID(property);
	return &(m.getPropf4((AttributeValues::Float4Property)id)).x;
}


void 
PhysicsPropertyManager::setMaterialFloatProperty(const std::string &material, const std::string &property, float value) {

	if (!m_PhysicsManager)
		m_PhysicsManager = PhysicsManager::GetInstance();

	PhysicsMaterial &m = m_PhysicsManager->getMaterial(material);
	int id = m.getAttribSet()->getID(property);
	m_PhysicsManager->getMaterial(material).setPropf((AttributeValues::FloatProperty)id, value);
}


void 
PhysicsPropertyManager::setMaterialVec4Property(const std::string &material, const std::string &property, float *value) {

	if (!m_PhysicsManager)
		m_PhysicsManager = PhysicsManager::GetInstance();

	PhysicsMaterial &m = m_PhysicsManager->getMaterial(material);
	int id = m.getAttribSet()->getID(property);
	vec4 v(value);
	m_PhysicsManager->getMaterial(material).setPropf4((AttributeValues::Float4Property)id, v);
}


float 
PhysicsPropertyManager::getGlobalFloatProperty(const std::string &property) {

	if (!m_PhysicsManager)
		m_PhysicsManager = PhysicsManager::GetInstance();

	int id = m_PhysicsManager->getAttribSet()->getID(property);
	return m_PhysicsManager->getPropf((AttributeValues::FloatProperty)id);
}


float *
PhysicsPropertyManager::getGlobalVec4Property(const std::string &property) {

	if (!m_PhysicsManager)
		m_PhysicsManager = PhysicsManager::GetInstance();

	int id = m_PhysicsManager->getAttribSet()->getID(property);
	return &(m_PhysicsManager->getPropf4((AttributeValues::Float4Property)id).x);
}


void 
PhysicsPropertyManager::setGlobalFloatProperty(const std::string &property, float value) {

	if (!m_PhysicsManager)
		m_PhysicsManager = PhysicsManager::GetInstance();

	int id = m_PhysicsManager->getAttribSet()->getID(property);
	m_PhysicsManager->setPropf((AttributeValues::FloatProperty)id, value);
}


void 
PhysicsPropertyManager::setGlobalVec4Property(const std::string &property, float *value) {

	if (!m_PhysicsManager)
		m_PhysicsManager = PhysicsManager::GetInstance();

	int id = m_PhysicsManager->getAttribSet()->getID(property);
	vec4 v(value);
	m_PhysicsManager->setPropf4((AttributeValues::Float4Property)id, v);
}
