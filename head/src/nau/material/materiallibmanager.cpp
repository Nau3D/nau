#include <nau/material/materiallibmanager.h>

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

		m_LibManager.begin()->second->clear();
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
	return getLib (aLibrary)->hasMaterial (name);
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


std::vector<std::string>*
MaterialLibManager::getLibNames()
{
	std::map<std::string, MaterialLib*>::iterator iter; /***MARK***/
	std::vector<std::string> *names = new std::vector<std::string>;

    for(iter = m_LibManager.begin(); iter != m_LibManager.end(); ++iter)
    {
        names->push_back(iter->first);   
    }
	return (names);
}


std::vector<std::string>*
MaterialLibManager::getMaterialNames(const std::string &lib) 
{
	return (getLib (lib)->getMaterialNames());
}


int 
MaterialLibManager::getNumLibs (void) 
{
	// FIXME: Type mismatch. size_t can be larger than an int
	return (int)(m_LibManager.size());
}

 
void
MaterialLibManager::addMaterial (std::string aLibrary, nau::material::Material* aMaterial)
{
	(getLib (aLibrary)->addMaterial (aMaterial));
}





//void CMaterialLibManager::addLib(CMaterialLib *mlib) {
//
//	m_libManager[mlib->m_filename] = mlib;
//}

//bool
//MaterialLibManager::validMatName(std::string lib, std::string matName) {
//
//	if (matName.empty())
//		return(INVALID_NAME);
//
//	int loc = matName.find( " ", 0 );
//	if( loc == 0 )
//		return(INVALID_NAME);
//
//	CMaterialLib *ml;
//
//	ml = getLib(lib);
//	int count = ml->lib.count(matName);
//
//	if (count != 0)
//		return(NAME_EXISTS);
//
//	return (OK);
//	
//}


//void 
//MaterialLibManager::save(std::string path) {
//
//	std::map<std::string, CMaterialLib*>::iterator iter;
//	std::vector<std::string> *names = new std::vector<std::string>;
//
//    for(iter = m_libManager.begin(); iter != m_libManager.end(); ++iter)
//    {
//		if (iter->first.substr(0,1) != " ")
//			m_libManager[iter->first]->save(CProject::Instance()->m_path);
//    }
//}
//
//
//
//void 
//MaterialLibManager::load(std::string filename, std::string path) {
//	
//	CMaterialLib *ml;
//	std::string libName = CFilename::RemoveExt(filename);
//	std::string fullName = path + "/" + filename;
//
//	ml = new CMaterialLib();
//	ml->load(fullName);
//
//	m_libManager[libName] = ml;
//}


//void 
//MaterialLibManager::addOwnMaterials ()
//{
//	Material *mat = new Material;
//	mat->getColor().setDiffuse( 0.8f, 0.8f, 0.8f, 1.0f );
//	mat->getColor().setEmission(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("Light Grey");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getColor().setDiffuse( 0.8f, 0.8f, 0.8f, 1.0f );
//	mat->getColor().setEmission(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("quad");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getColor().setEmission( 1.0f, 1.0f, 1.0f, 1.0f );
//	mat->getColor().setDiffuse(1.0f, 1.0f, 1.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("Emission White");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getColor().setEmission(1.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setDiffuse(1.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("Emission Red");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getColor().setEmission(0.0f, 1.0f, 0.0f, 1.0f);
//	mat->getColor().setDiffuse(0.0f, 1.0f, 0.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("Emission Green");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getColor().setEmission(0.0f, 0.0f, 1.0f, 1.0f);
//	mat->getColor().setDiffuse(0.0f, 0.0f, 1.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("Emission Blue");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getColor().setEmission(0.0f, 1.0f, 1.0f, 1.0f);
//	mat->getColor().setDiffuse(0.0f, 1.0f, 1.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("Emission Cyan");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getColor().setEmission(1.0f, 0.0f, 1.0f, 1.0f);
//	mat->getColor().setDiffuse(1.0f, 0.0f, 1.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("Emission Purple");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getColor().setEmission(1.0f, 1.0f, 0.0f, 1.0f);
//	mat->getColor().setDiffuse(1.0f, 1.0f, 0.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("Emission Yellow");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getColor().setEmission(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setDiffuse(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setAmbient(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->getColor().setSpecular(0.0f, 0.0f, 0.0f, 1.0f);
//	mat->setName("Black");
//	m_DefaultLib->addMaterial(mat);
//
//	mat = new Material;
//	mat->getState()->setProp(IState::ORDER, -1);
//	mat->setName("No Render");
//	m_DefaultLib->addMaterial(mat);
//}
