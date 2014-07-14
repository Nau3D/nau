#include <nau/material/materialgroup.h>
#include <nau/render/vertexdata.h>
#include <nau/render/irenderer.h>

#include <nau/math/vec3.h>
#include <nau/clogger.h>

#include <set>

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

/*
int 
MaterialGroup::getMaterialId()
{
   return m_MaterialId;
}

void 
MaterialGroup::setMaterialId (int id)
{
   this->m_MaterialId = id;
}
*/
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

void 
MaterialGroup::setIndexList (std::vector<unsigned int>* indices) /***MARK***/
{
	if (0 == m_IndexData) {
		m_IndexData = IndexData::create();
	}
	m_IndexData->setIndexData (indices);
	//m_VertexData->compile(); /***MARK***/
}

unsigned int
MaterialGroup::getNumberOfPrimitives(void) {

	unsigned int indexes = m_IndexData->getIndexSize();
	unsigned int primitive = m_Parent->getDrawingPrimitive();
	switch(primitive) {
	
		case nau::render::IRenderer::TRIANGLES_ADJACENCY: 
			return (indexes / 6);
		case nau::render::IRenderer::TRIANGLES: 
			return (indexes / 3);
		case nau::render::IRenderer::TRIANGLE_STRIP: 
		case nau::render::IRenderer::TRIANGLE_FAN:
			return (indexes - 2);
		case nau::render::IRenderer::LINES:
			return (indexes / 2);
		default: 
			return (indexes / 3);
	}

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

//// FIXME: This has to go away
//// see also: cworldfactory.cpp
//void 
//MaterialGroup::bakeMayaUVTextureProfile (float repeat_u, float repeat_v,
//				       float coverage_u, float coverage_v)
//{  
//  // Compute fudge factors
//  float FudgeU = repeat_u / coverage_u;
//  float FudgeV = repeat_v / coverage_v;
//
//  VertexData &aVertexData = m_Parent->getVertexData();
//
//  // Get parent Texcoords
//  std::vector<VertexData::Attr> &ParentTexCoordsList = aVertexData.getDataOf (VertexData::getAttribIndex("texCoord0"));
// 
//  std::vector<unsigned int>& IndexList = m_IndexData->getIndexData();
//
//  // Iterate index list and bake the UV fudge factors.
//  std::vector<unsigned int>::const_iterator index = IndexList.begin();
//  
//  std::set<int> UniqueIndices;
//
//  for (; index != IndexList.end (); index++){
//	UniqueIndices.insert (*index);
//  }
//
//  std::set<int>::const_iterator setIndex = UniqueIndices.begin();
//
//  for (; setIndex != UniqueIndices.end(); setIndex++) {
//    ParentTexCoordsList[*setIndex].x *= FudgeU;
//    ParentTexCoordsList[*setIndex].y *= FudgeV;
//
//	//LOG_ERROR ("UV(%f, %f)", ParentTexCoordsList[*setIndex].x, ParentTexCoordsList[*setIndex].y);
//  }
//}
