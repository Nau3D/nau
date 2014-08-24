
#include <nau.h>
#include <nau/material/materialgroup.h>
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


MaterialGroup::~MaterialGroup()
{
	delete m_IndexData;
}


const std::string& 
MaterialGroup::getMaterialName ()
{
	return m_MaterialName;
}


void 
MaterialGroup::setMaterialName (std::string name)
{
	this->m_MaterialName = name;
}


IndexData&
MaterialGroup::getIndexData (void)
{
	if (0 == m_IndexData) {
		m_IndexData = IndexData::create();
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
		m_IndexData = IndexData::create();
	}
	m_IndexData->setIndexData (indices);
}


unsigned int
MaterialGroup::getNumberOfPrimitives(void) {

	return RENDERER->getNumberOfPrimitives(this);
}


void 
MaterialGroup::setParent (IRenderable* parent)
{
	this->m_Parent = parent;
}


IRenderable& 
MaterialGroup::getParent ()
{
	return *(this->m_Parent);
}

