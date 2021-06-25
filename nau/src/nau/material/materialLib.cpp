#include "nau/material/materialLib.h"

#include "nau.h"

using namespace nau::material;

MaterialLib::MaterialLib (const std::string &libName) : 
  m_MaterialLib(),
  m_LibName (libName) {

	EVENTMANAGER->addListener("SHADER_CHANGED", this);
}

MaterialLib::~MaterialLib() {

	EVENTMANAGER->removeListener("SHADER_CHANGED", this);
	while (!m_MaterialLib.empty()) {
		m_MaterialLib.erase(m_MaterialLib.begin());
	}

}


void
MaterialLib::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<nau::event_::IEventData> &evt) {

	std::string *str;
	std::string s = "";

	str = (std::string *)evt->getData();

	if (*str == "Linked" && eventType == "SHADER_CHANGED") {

		for (auto m : m_MaterialLib) {

			if (m.second->getProgramName() == sender) {
				m.second->checkProgramValuesAndUniforms(s);
			}
		}
		if (s.size()) {
			std::shared_ptr<nau::event_::IEventData> e = nau::event_::EventFactory::Create("String");
			e->setData(&s);
			EVENTMANAGER->notifyEvent("MATERIAL_SHADER_USAGE_REPORT", sender, "", e);
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
		m_MaterialLib.erase(m_MaterialLib.begin());
	}
}


bool 
MaterialLib::hasMaterial (const std::string &materialName) {

	std::map<std::string, std::shared_ptr<Material>>::const_iterator mat = m_MaterialLib.find (materialName);
  
	return (m_MaterialLib.end() != mat);
}


std::shared_ptr<Material> &
MaterialLib::getMaterial (const std::string &materialName) {

	if (m_MaterialLib.count(materialName)) {
		return m_MaterialLib[materialName];
	}
	else
		return p_Default;
}


void
MaterialLib::getMaterialNames(const std::string &aName, std::vector<std::string>* ret) {

	bool wildCard = false;
	size_t len = aName.size();
	if (aName[len - 1] == '*') {
		wildCard = true;
		len--;
	}

	for (auto& mat : m_MaterialLib) {
		if (wildCard && 0 == aName.substr(0,len).compare(mat.first.substr(0,len)))
			ret->push_back(mat.first); 
		else if (0 == aName.substr(0, len).compare(mat.first))
			ret->push_back(mat.first);
	}
}


unsigned int
MaterialLib::getMaterialCount() {

	return (unsigned int)m_MaterialLib.size();
}

void
MaterialLib::getMaterialNames(std::vector<std::string>* ret) {

	for(auto& mat:m_MaterialLib) {
		ret->push_back(mat.first);   
	}
}


void 
MaterialLib::addMaterial (std::shared_ptr<Material> &aMaterial) {

	std::string matName = aMaterial->getName();
	std::shared_ptr<Material> &pCurrentMat = getMaterial (matName);

	// add if it does not exist
	if (!pCurrentMat) {
		m_MaterialLib[matName] = aMaterial;
	} 
}


