#include <nau/material/materialgroup.h>

#include <nau.h>
#include <nau/render/vertexdata.h>
#include <nau/render/irenderer.h>
#include <nau/math/vec3.h>
#include <nau/clogger.h>


using namespace nau::material;
using namespace nau::render;
using namespace nau::math;


MaterialGroup::MaterialGroup() :
//	m_MaterialId (0),
	m_Parent (0),
	m_MaterialName ("default"),
	m_IndexData (0)
{
   //ctor
}


MaterialGroup::MaterialGroup(IRenderable *parent, std::string materialName) :
//	m_MaterialId (0),
m_Parent(parent),
m_MaterialName(materialName),
m_IndexData(0)
{
	//ctor
}


MaterialGroup::~MaterialGroup()
{
	delete m_IndexData;
}


void
MaterialGroup::setParent(IRenderable* parent)
{
	this->m_Parent = parent;
}


void 
MaterialGroup::setMaterialName (std::string name)
{
	this->m_MaterialName = name;
	m_IndexData->setName(getName());
}


std::string &
MaterialGroup::getName() {

	m_Name = m_Parent->getName() + ":" + m_MaterialName;
	return m_Name;
}


const std::string&
MaterialGroup::getMaterialName ()
{
	return m_MaterialName;
}


IndexData&
MaterialGroup::getIndexData (void)
{
	if (0 == m_IndexData) {
		m_IndexData = IndexData::create(getName());
	}
	return (*m_IndexData);
}


unsigned int
MaterialGroup::getIndexOffset(void)
{
	return 0;
}


unsigned int
MaterialGroup::getIndexSize(void)
{
	if (0 == m_IndexData) {
		return 0;
	}
	return m_IndexData->getIndexSize();
}


void 
MaterialGroup::setIndexList (std::vector<unsigned int>* indices) 
{
	if (0 == m_IndexData) {
		m_IndexData = IndexData::create(getName());
	}
	m_IndexData->setIndexData (indices);
}


unsigned int
MaterialGroup::getNumberOfPrimitives(void) {

	return RENDERER->getNumberOfPrimitives(this);
}



IRenderable& 
MaterialGroup::getParent ()
{
	return *(this->m_Parent);
}


void 
MaterialGroup::updateIndexDataName() {

	m_IndexData->setName(getName());
}