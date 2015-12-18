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

}


void 
Mesh::eventReceived(const std::string &sender,
	const std::string &eventType,
	const std::shared_ptr<IEventData> &evt) {

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


std::shared_ptr<nau::geometry::VertexData>&
Mesh::getVertexData (void) {

	if (!m_VertexData) {
		m_VertexData = VertexData::Create(m_Name);
	}
	return (m_VertexData);
}


std::shared_ptr<nau::geometry::IndexData>&
Mesh::getIndexData() {

	if (!m_IndexData)
		m_IndexData = IndexData::Create(m_Name);

	if (!m_IndexData->getIndexData())
		createUnifiedIndexVector();

	return m_IndexData;
}


std::vector<std::shared_ptr<nau::material::MaterialGroup>>&
Mesh::getMaterialGroups (void) {

	return (m_vMaterialGroups);
}


void 
Mesh::getMaterialNames(std::set<std::string> *nameList) {

	assert(nameList != 0);

	for (auto &iter:m_vMaterialGroups) {
		const std::string& s = iter->getMaterialName();
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

		size_t size = m_VertexData->getDataOf(VertexData::GetAttribIndex(std::string("position")))->size();
		std::shared_ptr<std::vector<VertexData::Attr>> idsArray = 
			std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(size));

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

	size_t size = m_VertexData->getDataOf(VertexData::GetAttribIndex(std::string("position")))->size();
	std::vector<int> idsArray = std::vector<int>(size, -1);
	std::vector<int> outlaws;

	createUnifiedIndexVector();
	std::vector<unsigned int> &unified = *(m_IndexData->getIndexData().get());
	unsigned int index0, index1, index2, aux0, aux1, aux2;
	for (unsigned int i = 0 ; i < getNumberOfVertices()/3; i++) {
	
		index0 = unified[ i * 3 ];
		index1 = unified[ i * 3 + 1 ];
		index2 = unified[ i * 3 + 2 ];

		if (idsArray[index2] == -1)
			idsArray[index2] = i;
		else if (idsArray[index1] == -1) {
			idsArray[index1] = i;
			aux0 = unified[ i * 3 ];
			aux1 = unified[ i * 3 + 1];
			aux2 = unified[ i * 3 +2 ];
			unified[ i * 3 ]    = aux2;
			unified[ i * 3 + 1] = aux0;
			unified[ i * 3 + 2] = aux1;
		}
		else if (idsArray[index0] == -1) {
			idsArray[index0] = i;
			aux0 = unified[ i * 3 ];
			aux1 = unified[ i * 3 + 1];
			aux2 = unified[ i * 3 +2 ];
			unified[ i * 3 ]    = aux1;
			unified[ i * 3 + 1] = aux2;
			unified[ i * 3 + 2] = aux0;
		}
		else {
			m_VertexData->appendVertex(index2);
			unified[i * 3 + 2] = (unsigned int)size;
			size++;
		}
	}

	// Copy back from UnifiedIndex to MaterialGroups.index
	size_t base = 0;
	std::vector<unsigned int>::iterator indexIter;
	indexIter = m_IndexData->getIndexData()->begin();
	for (auto &iter:m_vMaterialGroups) {

		size_t size = iter->getIndexData()->getIndexData()->size();
		std::shared_ptr<std::vector<unsigned int>> matGroupIndexes = 
			std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>(size));
		matGroupIndexes->assign(indexIter+base, indexIter+(base+size));
		iter->getIndexData()->setIndexData(matGroupIndexes);
		base += size;
	}
}


unsigned int 
Mesh::getNumberOfVertices (void) {

	return (int)(getVertexData()->getDataOf (VertexData::GetAttribIndex(std::string("position"))))->size();
}


void 
Mesh::createUnifiedIndexVector() {

	std::shared_ptr<std::vector<unsigned int>> unified = 
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>);

	for (auto &iter: m_vMaterialGroups) {

		std::shared_ptr<std::vector<unsigned int>> &matGroupIndexes = iter->getIndexData()->getIndexData();
		unified->insert(unified->end(), matGroupIndexes->begin(),matGroupIndexes->end());
	}
	m_IndexData->setIndexData(unified);
}


void 
Mesh::addMaterialGroup (std::shared_ptr<MaterialGroup> &materialGroup, int offset) {

	/*
	- search material in vector
	- if it doesn't exist push back
	- if exists merge them
	*/
	bool found = false;

	for (auto &aMaterialGroup :m_vMaterialGroups ) {

		if (aMaterialGroup->getMaterialName() == materialGroup->getMaterialName()){
			std::shared_ptr<nau::geometry::IndexData> &indexVertexData = aMaterialGroup->getIndexData();

			indexVertexData->add (materialGroup->getIndexData());
			found = true;
			break;
		}
	}
	if (!found) {
		std::shared_ptr<MaterialGroup> newMat = MaterialGroup::Create(this, materialGroup->getMaterialName());

		newMat->getIndexData()->add (materialGroup->getIndexData());
		m_vMaterialGroups.push_back (newMat);		
	}
}


void 
Mesh::addMaterialGroup (std::shared_ptr<MaterialGroup> &materialGroup, IRenderable *aRenderable) {

	/* In this case it is necessary to copy the vertices from the 
	 * IRenderable into the local buffer and reindex the materialgroup
	 */
	std::shared_ptr<VertexData> &renderableVertexData = aRenderable->getVertexData();

	std::shared_ptr<VertexData> newData = VertexData::Create("dummy");

	std::shared_ptr<std::vector<VertexData::Attr>> list[VertexData::MaxAttribs], poolList[VertexData::MaxAttribs];

	for (int i = 0; i < VertexData::MaxAttribs; i++) {
		list[i] = std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>);
	}
	std::map<unsigned int, unsigned int> newIndicesMap;

	std::shared_ptr<std::vector<unsigned int>>& indices	= materialGroup->getIndexData()->getIndexData();
	std::vector<unsigned int>::iterator indexesIter;
	
	indexesIter = indices->begin();

	for (int i = 0 ; i < VertexData::MaxAttribs; i++)
		poolList[i] = renderableVertexData->getDataOf(i);

	for ( ; indexesIter != indices->end(); indexesIter++) {

		if (0 == newIndicesMap.count ((*indexesIter))) {

			for (int i = 0; i < VertexData::MaxAttribs; i++) {
				if (poolList[i]->size()) 
					list[i]->push_back(poolList[i]->at((*indexesIter)));
			}


			newIndicesMap[(*indexesIter)] = (unsigned int)(list[0]->size() - 1);
		} 
		(*indexesIter) = newIndicesMap[(*indexesIter)];
	}


	for ( int i = 0; i < VertexData::MaxAttribs; i++) 
		newData->setAttributeDataFor(i,list[i]);

	int offset = getVertexData()->add (newData);
	
	materialGroup->getIndexData()->offsetIndices (offset);
	addMaterialGroup (materialGroup);			
}


void 
Mesh::merge (nau::render::IRenderable *aRenderable) {

	std::shared_ptr<VertexData> &vVertexData = aRenderable->getVertexData();

	int ofs = getVertexData()->add (vVertexData);

	std::vector<std::shared_ptr<nau::material::MaterialGroup>> &materialGroups = aRenderable->getMaterialGroups();


	for (auto &aMaterialGroup : materialGroups) {

		std::shared_ptr<IndexData> &indexData = aMaterialGroup->getIndexData();
		indexData->offsetIndices (ofs);

		addMaterialGroup (aMaterialGroup);
	}
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

