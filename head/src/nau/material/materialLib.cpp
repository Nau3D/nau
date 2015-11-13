#include "nau/material/materialLib.h"

#include "nau.h"

using namespace nau::material;

MaterialLib::MaterialLib (std::string libName) : 
  m_MaterialLib(),
  m_LibName (libName) {

	EVENTMANAGER->addListener("SHADER_CHANGED", this);
}

MaterialLib::~MaterialLib() {

   //dtor
	EVENTMANAGER->removeListener("SHADER_CHANGED", this);
	while (!m_MaterialLib.empty()) {
		delete ((*m_MaterialLib.begin()).second);
		m_MaterialLib.erase(m_MaterialLib.begin());
	}

}


void
MaterialLib::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt) {

	std::string *str;
		str = (std::string *)evt->getData();

	if (*str == "Linked" && eventType == "SHADER_CHANGED") {

		for (auto m : m_MaterialLib) {

			if (m.second->getProgramName() == sender)
				m.second->checkProgramValuesAndUniforms();
		}
	}
}


std::string &
MaterialLib::getName() {

	return(m_LibName);
}


void
MaterialLib::clear() {

	while (!m_MaterialLib.empty()) {

		delete m_MaterialLib.begin()->second;
		m_MaterialLib.erase(m_MaterialLib.begin());
	}
}


bool 
MaterialLib::hasMaterial (std::string MaterialName) {

	std::map<std::string, Material*>::const_iterator mat = m_MaterialLib.find (MaterialName);
  
	return (m_MaterialLib.end() != mat);
}


Material*
MaterialLib::getMaterial (std::string MaterialName) {

	std::map<std::string, Material*>::const_iterator mat = m_MaterialLib.find (MaterialName);

	if (m_MaterialLib.end() == mat) {
		return &p_Default;
	}

	return (mat->second);
}


void
MaterialLib::getMaterialNames(const std::string &aName, std::vector<std::string>* ret) {

	size_t len = aName.size();
	if (aName[len-1] == '*')
		len--;

	for (auto& mat : m_MaterialLib) {
		if (0 == aName.substr(0,len).compare(mat.first.substr(0,len)))
			ret->push_back(mat.first);   
	}
}


void 
MaterialLib::getMaterialNames(std::vector<std::string>* ret) {

	for(auto& mat:m_MaterialLib) {
		ret->push_back(mat.first);   
	}
}


void 
MaterialLib::addMaterial (nau::material::Material* aMaterial) {

	std::string MatName(aMaterial->getName());
	Material *pCurrentMat = getMaterial (MatName);
  
	if (pCurrentMat != 0 && pCurrentMat != aMaterial) {

	}
	
	if (pCurrentMat->getName() != aMaterial->getName()) { 
		m_MaterialLib[MatName] = aMaterial;
	} 
	else if (pCurrentMat != aMaterial) {
		// TODO: Add to log
		//std::cout << "Matlib: adding different valid pointers for same name, possible memleak!" << std::endl;
	}
}


