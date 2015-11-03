#include "nau/geometry/mesh.h"


#include "nau.h"
#include "nau/material/materialGroup.h"
#include "nau/math/vec3.h"
#include "nau/render/iRenderable.h"
#include "nau/geometry/vertexData.h"


using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;
using namespace nau::math;


Mesh *
Mesh::createUnregisteredMesh() {

	return new Mesh();
}


Mesh::Mesh(void) :
	m_VertexData (0),
	m_IndexData(0),
	m_vMaterialGroups (),
	m_DrawPrimitive(nau::render::IRenderable::TRIANGLES),
	m_Name (""),
	m_NumberOfPrimitives(-1)
{
	m_RealDrawPrimitive = RENDERER->translateDrawingPrimitive(m_DrawPrimitive);
}


Mesh::~Mesh(void) {

	if (0 != m_VertexData) {
		delete m_VertexData;
		m_VertexData = 0;
	}

	if (0 != m_IndexData) {
		delete m_IndexData;
		m_IndexData = 0;
	}

	std::vector<nau::material::MaterialGroup*>::iterator matIter;

	matIter = m_vMaterialGroups.begin();
	
	while (!m_vMaterialGroups.empty()){
		delete((*m_vMaterialGroups.begin()));
		m_vMaterialGroups.erase(m_vMaterialGroups.begin());
	}
}


void 
Mesh::setName (std::string name) {

	m_Name = name;
	if (m_VertexData)
		m_VertexData->setName(name);

	for (auto m : m_vMaterialGroups)
		m->updateIndexDataName();
}


std::string& 
Mesh::getName (void) {

	return m_Name;
}


unsigned int
Mesh::getDrawingPrimitive() {

	return(m_DrawPrimitive);
}


unsigned int
Mesh::getRealDrawingPrimitive() {

	return(m_RealDrawPrimitive);
}


void
Mesh::setDrawingPrimitive(unsigned int aDrawingPrimitive) {

	m_DrawPrimitive = aDrawingPrimitive;
	m_RealDrawPrimitive = RENDERER->translateDrawingPrimitive(m_DrawPrimitive);
}


void 
Mesh::setNumberOfVerticesPerPatch(int i) {

	m_VerticesPerPatch = i;
}


int
Mesh::getnumberOfVerticesPerPatch() {

	return m_VerticesPerPatch;

}


VertexData& 
Mesh::getVertexData (void) {

	if (0 == m_VertexData) {
		m_VertexData = VertexData::create(m_Name);
	}
	return (*m_VertexData);
}


IndexData&
Mesh::getIndexData() {

	if (m_UnifiedIndex.size() == 0)
		createUnifiedIndexVector();

	if (!m_IndexData)
		m_IndexData = IndexData::create(m_Name);

	m_IndexData->setIndexData(&m_UnifiedIndex);
	return *m_IndexData;
}


std::vector<nau::material::MaterialGroup*>&
Mesh::getMaterialGroups (void) {

	return (m_vMaterialGroups);
}


void 
Mesh::getMaterialNames(std::set<std::string> *nameList) {

	assert(nameList != 0);

	std::vector<nau::material::MaterialGroup*>::iterator iter;

	iter = m_vMaterialGroups.begin();

	for ( ; iter != m_vMaterialGroups.end(); ++iter) {
		const std::string& s = (*iter)->getMaterialName();
		nameList->insert(s);
	}
}


void 
Mesh::prepareTriangleIDs(unsigned int sceneObjectID) {

	if (m_DrawPrimitive != IRenderable::TRIANGLES)
		return;

	if (sceneObjectID != 0) {
	
		prepareIndexData();
		createUnifiedIndexVector();

		size_t size = m_VertexData->getDataOf(VertexData::GetAttribIndex(std::string("position"))).size();
		std::vector<VertexData::Attr>* idsArray = new std::vector<VertexData::Attr>(size);

		int primitiveOffset = 3;//getPrimitiveOffset();
		for (unsigned int i = 0; i < size; i++) {
			idsArray->at(i).x = (float)sceneObjectID;
			idsArray->at(i).y = (float)(i / primitiveOffset);
		}
		m_VertexData->setAttributeDataFor(VertexData::GetAttribIndex(std::string("triangleID")), idsArray);
	}
}


void 
Mesh::prepareIndexData() {

	size_t size = m_VertexData->getDataOf(VertexData::GetAttribIndex(std::string("position"))).size();
	std::vector<int> idsArray = std::vector<int>(size, -1);
	std::vector<int> outlaws;

	createUnifiedIndexVector();

	unsigned int index0, index1, index2, aux0, aux1, aux2;
	for (unsigned int i = 0 ; i < getNumberOfVertices()/3; i++) {
	
		index0 = m_UnifiedIndex[ i * 3 ];
		index1 = m_UnifiedIndex[ i * 3 + 1 ];
		index2 = m_UnifiedIndex[ i * 3 + 2 ];

		if (idsArray[index2] == -1)
			idsArray[index2] = i;
		else if (idsArray[index1] == -1) {
			idsArray[index1] = i;
			aux0 = m_UnifiedIndex[ i * 3 ];
			aux1 = m_UnifiedIndex[ i * 3 + 1];
			aux2 = m_UnifiedIndex[ i * 3 +2 ];
			m_UnifiedIndex[ i * 3 ]    = aux2;
			m_UnifiedIndex[ i * 3 + 1] = aux0;
			m_UnifiedIndex[ i * 3 + 2] = aux1;
		}
		else if (idsArray[index0] == -1) {
			idsArray[index0] = i;
			aux0 = m_UnifiedIndex[ i * 3 ];
			aux1 = m_UnifiedIndex[ i * 3 + 1];
			aux2 = m_UnifiedIndex[ i * 3 +2 ];
			m_UnifiedIndex[ i * 3 ]    = aux1;
			m_UnifiedIndex[ i * 3 + 1] = aux2;
			m_UnifiedIndex[ i * 3 + 2] = aux0;
		}
		else {
			m_VertexData->appendVertex(index2);
			m_UnifiedIndex[i * 3 + 2] = (unsigned int)size;
			size++;
		}
	}

	// Copy back from UnifiedIndex to MaterialGroups.index
	std::vector<nau::material::MaterialGroup*>::iterator iter;

	iter = m_vMaterialGroups.begin();
	size_t base = 0;
	std::vector<unsigned int>::iterator indexIter;
	indexIter = m_UnifiedIndex.begin();
	for ( ; iter != m_vMaterialGroups.end(); iter ++) {

		size_t size = (*iter)->getIndexData().getIndexData().size();
		std::vector<unsigned int>* matGroupIndexes = new std::vector<unsigned int>(size);
		matGroupIndexes->assign(indexIter+base, indexIter+(base+size));
		(*iter)->getIndexData().setIndexData(matGroupIndexes);
		base += size;
	}
}


unsigned int 
Mesh::getNumberOfVertices (void) {

	return (int)(getVertexData().getDataOf (VertexData::GetAttribIndex(std::string("position")))).size();
}


void 
Mesh::createUnifiedIndexVector() {

	m_UnifiedIndex.clear();

	std::vector<nau::material::MaterialGroup*>::iterator iter;

	iter = m_vMaterialGroups.begin();
	for ( ; iter != m_vMaterialGroups.end(); iter ++) {

		std::vector<unsigned int> matGroupIndexes = (*iter)->getIndexData().getIndexData();
		m_UnifiedIndex.insert(m_UnifiedIndex.end(), matGroupIndexes.begin(),matGroupIndexes.end());
	}
}


void 
Mesh::addMaterialGroup (MaterialGroup* materialGroup, int offset) {

	/*
	- search material in vector
	- if it doesn't exist push back
	- if exists merge them
	*/
	std::vector<MaterialGroup*>::iterator matGroupIter;

	matGroupIter = m_vMaterialGroups.begin();

	for ( ; matGroupIter != m_vMaterialGroups.end(); matGroupIter++ ) {
		MaterialGroup* aMaterialGroup = (*matGroupIter);

		if (aMaterialGroup->getMaterialName() == materialGroup->getMaterialName()){
			IndexData &indexVertexData = aMaterialGroup->getIndexData();

			indexVertexData.add (materialGroup->getIndexData());
			break;
		}
	}
	if (m_vMaterialGroups.end() == matGroupIter) {
		MaterialGroup *newMat = MaterialGroup::Create(this, materialGroup->getMaterialName());

		//newMat->setMaterialName (materialGroup->getMaterialName());
		//newMat->setParent (this);	
		newMat->getIndexData().add (materialGroup->getIndexData());
		m_vMaterialGroups.push_back (newMat);		
	}
}


void 
Mesh::addMaterialGroup (MaterialGroup* materialGroup, IRenderable *aRenderable) {

	/* In this case it is necessary to copy the vertices from the 
	 * IRenderable into the local buffer and reindex the materialgroup
	 */
	VertexData &renderableVertexData = aRenderable->getVertexData();

	VertexData *newData = VertexData::create("dummy"); 

	std::vector<VertexData::Attr> *list[VertexData::MaxAttribs], poolList[VertexData::MaxAttribs];

	for (int i = 0; i < VertexData::MaxAttribs; i++) {
		list[i] = new  std::vector<VertexData::Attr>;
	}
	std::map<unsigned int, unsigned int> newIndicesMap;

	std::vector<unsigned int>& indices	= materialGroup->getIndexData().getIndexData();
	std::vector<unsigned int>::iterator indexesIter;
	
	indexesIter = indices.begin();

	for (int i = 0 ; i < VertexData::MaxAttribs; i++)
		poolList[i] = renderableVertexData.getDataOf(i);

	for ( ; indexesIter != indices.end(); indexesIter++) {

		if (0 == newIndicesMap.count ((*indexesIter))) {

			for (int i = 0; i < VertexData::MaxAttribs; i++) {
				if (poolList[i].size()) 
					list[i]->push_back(poolList[i].at((*indexesIter)));
			}


			newIndicesMap[(*indexesIter)] = (unsigned int)(list[0]->size() - 1);
		} 
		(*indexesIter) = newIndicesMap[(*indexesIter)];
	}


	for ( int i = 0; i < VertexData::MaxAttribs; i++) 
		newData->setAttributeDataFor(i,list[i]);

	int offset = getVertexData().add (*newData);
	delete newData;
	
	materialGroup->getIndexData().offsetIndices (offset);
	addMaterialGroup (materialGroup);			
}


void 
Mesh::merge (nau::render::IRenderable *aRenderable) {

	VertexData &vVertexData = aRenderable->getVertexData();

	int ofs = getVertexData().add (vVertexData);

	std::vector<MaterialGroup*> &materialGroups = aRenderable->getMaterialGroups();
	std::vector<MaterialGroup*>::iterator materialIter;

	materialIter = materialGroups.begin();

	for ( ; materialIter != materialGroups.end(); materialIter++) {
		MaterialGroup *aMaterialGroup = (*materialIter);
		IndexData &indexData = aMaterialGroup->getIndexData();
		indexData.offsetIndices (ofs);

		addMaterialGroup (aMaterialGroup);
	}
//	delete aRenderable;
}


std::string 
Mesh::getType (void) {

	return "Mesh";
}


void 
Mesh::unitize(vec3 &center, vec3 &min, vec3 &max) {

	m_VertexData->unitize(center, min,max);
}


void
Mesh::resetCompilationFlags() {

	for (unsigned int i = 0; i < m_vMaterialGroups.size(); ++i) {

		m_vMaterialGroups[i]->resetCompilationFlag();
	}
}

