#include "nau/material/materialLibManager.h"

using namespace nau::material;


MaterialLibManager::MaterialLibManager() : 
	m_LibManager (),
	m_DefaultLib (getLib (DEFAULTMATERIALLIBNAME))
{
	//addOwnMaterials();
}


MaterialLibManager::~MaterialLibManager()
{
	while (!m_LibManager.empty()){
		delete((*m_LibManager.begin()).second);
		//m_LibManager.begin()->second->clear();
		m_LibManager.erase(m_LibManager.begin());
	}

}


void
MaterialLibManager::clear()
{
	while (!m_LibManager.empty()){
	
		m_LibManager.begin()->second->clear();
		m_LibManager.erase(m_LibManager.begin());
	}
	m_DefaultLib = getLib (DEFAULTMATERIALLIBNAME);
	//addOwnMaterials();

}


//

MaterialLib*
MaterialLibManager::getLib (std::string libName)
{
	if (m_LibManager.find (libName) == m_LibManager.end()) {
		m_LibManager[libName] = new MaterialLib (libName);
	}
	return (m_LibManager[libName]);
}


bool
MaterialLibManager::hasLibrary(std::string lib)
{
	if (m_LibManager.count(lib))
		return true;
	else
		return false;
}


bool 
MaterialLibManager::hasMaterial (std::string aLibrary, std::string name)
{
	return getLib(aLibrary)->hasMaterial (name);
}


Material*
MaterialLibManager::getDefaultMaterial (std::string materialName)
{
	return (m_DefaultLib->getMaterial (materialName));
}


Material*
MaterialLibManager::getMaterial (MaterialID &materialID)
{
	MaterialLib *ml;

	ml = getLib (materialID.getLibName());
	return ml->getMaterial (materialID.getMaterialName());
}


Material*
MaterialLibManager::getMaterial(std::string lib, std::string mat)
{
	MaterialLib *ml;

	ml = getLib (lib);
	return (ml->getMaterial (mat));
}


Material *
MaterialLibManager::createMaterial(std::string material) {

	Material *mat;

	Material *m;
	m = getMaterial(DEFAULTMATERIALLIBNAME, "dirLightDifAmbPix");
	if (m->getName() == "dirLightDifAmbPix")
		mat = m->clone();
	else 
		mat = new Material();

	mat->setName(material);
	addMaterial(DEFAULTMATERIALLIBNAME, mat);
	return mat;
}


Material *
MaterialLibManager::createMaterial(std::string library, std::string material) {

	Material *mat;

	Material *m;
	m = getMaterial(DEFAULTMATERIALLIBNAME, "dirLightDifAmbPix");
	if (m->getName() == "dirLightDifAmbPix")
		mat = m->clone();
	else 
		mat = new Material();
	mat->setName(material);
	addMaterial(library, mat);
	return mat;
}


void
MaterialLibManager::getLibNames(std::vector<std::string>* names) {

    for(auto &lib: m_LibManager) {
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
MaterialLibManager::addMaterial (std::string aLibrary, nau::material::Material* aMaterial) {

	(getLib (aLibrary)->addMaterial (aMaterial));
}


