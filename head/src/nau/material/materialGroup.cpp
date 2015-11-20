#include "nau/material/materialGroup.h"

#include "nau.h"
#include "nau/geometry/vertexData.h"
#include "nau/render/opengl/glMaterialGroup.h"
#include "nau/math/vec3.h"
#include "nau/clogger.h"


using namespace nau::material;
using namespace nau::render;
using namespace nau::render::opengl;
using namespace nau::math;


std::shared_ptr<MaterialGroup>
MaterialGroup::Create(nau::render::IRenderable *parent, std::string materialName) {

#ifdef NAU_OPENGL
	return std::shared_ptr<MaterialGroup>(new GLMaterialGroup(parent, materialName));
#endif
}


MaterialGroup::MaterialGroup() :
	m_Parent (0),
	m_MaterialName ("default") {
   //ctor
}


MaterialGroup::MaterialGroup(IRenderable *parent, std::string materialName) :
	m_Parent(parent),
	m_MaterialName(materialName) {
	//ctor
}


MaterialGroup::~MaterialGroup() {

}


void
MaterialGroup::setParent(IRenderable* parent) {

	this->m_Parent = parent;
}


void 
MaterialGroup::setMaterialName (std::string name) {

	this->m_MaterialName = name;
	m_IndexData->setName(getName());
}


std::string &
MaterialGroup::getName() {

	m_Name = m_Parent->getName() + ":" + m_MaterialName;
	return m_Name;
}


const std::string&
MaterialGroup::getMaterialName () {

	return m_MaterialName;
}


std::shared_ptr<nau::geometry::IndexData>&
MaterialGroup::getIndexData (void) {

	if (!m_IndexData) {
		m_IndexData = IndexData::Create(getName());
	}
	return (m_IndexData);
}


size_t
MaterialGroup::getIndexOffset(void) {

	return 0;
}


size_t
MaterialGroup::getIndexSize(void) {

	if (!m_IndexData) {
		return 0;
	}
	return m_IndexData->getIndexSize();
}


void 
MaterialGroup::setIndexList (std::shared_ptr<std::vector<unsigned int>> &indices) {

	if (!m_IndexData) {
		m_IndexData.reset();
		m_IndexData = IndexData::Create(getName());
	}
	m_IndexData->setIndexData (indices);
}


unsigned int
MaterialGroup::getNumberOfPrimitives(void) {

	return RENDERER->getNumberOfPrimitives(this);
}



IRenderable& 
MaterialGroup::getParent () {

	return *(this->m_Parent);
}


void 
MaterialGroup::updateIndexDataName() {

	m_IndexData->setName(getName());
}