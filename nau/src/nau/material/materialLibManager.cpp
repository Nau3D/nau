#include "nau/material/materialLibManager.h"

using namespace nau::material;


MaterialLibManager::MaterialLibManager() : 
	m_LibManager (),
	m_DefaultLib (getLib (DEFAULTMATERIALLIBNAME)) {

}


MaterialLibManager::~MaterialLibManager() {

	while (!m_LibManager.empty()){
		delete((*m_LibManager.begin()).second);
		m_LibManager.erase(m_LibManager.begin());
	}

}


void
MaterialLibManager::clear() {

	while (!m_LibManager.empty()){
	
		delete((*m_LibManager.begin()).second);
//		m_LibManager.begin()->second->clear();
		m_LibManager.erase(m_LibManager.begin());
	}
	m_DefaultLib = getLib (DEFAULTMATERIALLIBNAME);
	printf("Material Lib size: %d\n", m_LibManager.size());
}


MaterialLib*
MaterialLibManager::getLib (const std::string &libName)
{
	if (m_LibManager.find (libName) == m_LibManager.end()) {
		m_LibManager[libName] = new MaterialLib (libName);
	}
	return (m_LibManager[libName]);
}


bool
MaterialLibManager::hasLibrary(const std::string &libName) {

	if (m_LibManager.count(libName))
		return true;
	else
		return false;
}


bool 
MaterialLibManager::hasMaterial (const std::string &aLibrary, const std::string &name) {

	return getLib(aLibrary)->hasMaterial (name);
}


bool
MaterialLibManager::hasMaterial(const std::string &fullName) {


	std::string mat, lib;
	size_t pos = fullName.find_first_of(":");
	if (pos != std::string::npos) {
		lib = fullName.substr(0, pos);
		mat = fullName.substr(pos + 2, fullName.length());
	}
	else {
		mat = fullName;
		lib = DEFAULTMATERIALLIBNAME;
	}
	return getLib(lib)->hasMaterial(mat);
}


std::shared_ptr<Material> &
MaterialLibManager::getMaterialFromDefaultLib(const std::string &materialName) {

	return (m_DefaultLib->getMaterial (materialName));
}


std::shared_ptr<Material> &
MaterialLibManager::getMaterial (MaterialID &materialID) {

	MaterialLib *ml;

	ml = getLib (materialID.getLibName());
	return ml->getMaterial (materialID.getMaterialName());
}


std::shared_ptr<Material> &
MaterialLibManager::getMaterial(const std::string &lib, const std::string &mat) {

	return (getLib(lib)->getMaterial (mat));
}


std::shared_ptr<Material> &
MaterialLibManager::getMaterial(const std::string &fullMatName) {

	std::string mat, lib;
	size_t pos = fullMatName.find_first_of(":");
	if (pos != std::string::npos) {
		lib = fullMatName.substr(0, pos);
		mat = fullMatName.substr(pos + 2, fullMatName.length());
	}
	else {
		mat = fullMatName;
		lib = DEFAULTMATERIALLIBNAME;
	}
	return (getLib(lib)->getMaterial(mat));
}


std::shared_ptr<Material>
MaterialLibManager::createMaterial(const std::string &material) {

	std::shared_ptr<Material> mat;

	std::shared_ptr<Material> &m = getMaterial(DEFAULTMATERIALLIBNAME, "__nauDefault");
	if (m)
		mat = cloneMaterial(m);
	else 
		mat = std::shared_ptr<Material>(new Material());

	mat->setName(material);
	addMaterial(DEFAULTMATERIALLIBNAME, mat);
	return mat;
}


std::shared_ptr<Material>
MaterialLibManager::createMaterial(const std::string &library, const std::string &material) {

	std::shared_ptr<Material> mat;

	std::shared_ptr<Material> &m = getMaterial(DEFAULTMATERIALLIBNAME, "__nauDefault");
	if (m)
		mat = cloneMaterial(m);
	else 
		mat = std::shared_ptr<Material>(new Material());

	mat->setName(material);
	addMaterial(library, mat);

	return mat;
}


std::shared_ptr<Material> 
MaterialLibManager::cloneMaterial(std::shared_ptr<Material> &m) {

	return m->clone();
}


void
MaterialLibManager::getLibNames(std::vector<std::string>* names) {

    for(auto &lib: m_LibManager) {
        names->push_back(lib.first);   
    }
}

void 
MaterialLibManager::getNonEmptyLibNames(std::vector<std::string>* names) {

	for (auto &lib : m_LibManager) {
		if (lib.second->getMaterialCount())
			names->push_back(lib.first);
	}
}


void
MaterialLibManager::getMaterialNames(const std::string &lib, std::vector<std::string> *ret) {

	getLib(lib)->getMaterialNames(ret);
}


unsigned int
MaterialLibManager::getNumLibs (void) {

	return (unsigned int)(m_LibManager.size());
}

 
void
MaterialLibManager::addMaterial (const std::string &aLibrary, std::shared_ptr<Material> &aMaterial) {

	(getLib (aLibrary)->addMaterial (aMaterial));
}


