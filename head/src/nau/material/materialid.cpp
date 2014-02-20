#include <nau/material/materialid.h>
#include <nau.h>
#include <nau/material/materiallibmanager.h>

using namespace nau::material;

MaterialID::MaterialID (void):
	m_LibName(),
	m_MatName(),
	m_MatPtr(0)
{
}

MaterialID::MaterialID (std::string libName, std::string matName):
	m_LibName (libName), 
	m_MatName (matName)
{
	m_MatPtr = MATERIALLIBMANAGER->getMaterial(libName,matName);
}


MaterialID::~MaterialID() 
{
	
}

void 
MaterialID::setMaterialID (std::string libName, std::string matName)
{
	m_LibName = libName;
	m_MatName = matName;
	m_MatPtr = MATERIALLIBMANAGER->getMaterial(libName,matName);

}

Material*
MaterialID::getMaterialPtr()
{
	return m_MatPtr;
}

const std::string& 
MaterialID::getLibName (void)
{
	return m_LibName;
}

const std::string& 
MaterialID::getMaterialName (void)
{
	return m_MatName;
}

