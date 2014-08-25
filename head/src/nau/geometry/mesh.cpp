


#include <nau/geometry/mesh.h>
#include <nau/material/imaterialgroup.h>
#include <nau/material/materialgroup.h>
#include <nau/render/irenderable.h>
#include <nau/render/vertexdata.h>
#include <nau/math/vec3.h>
#include <nau.h>

//#include <algorithm>

using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;
using namespace nau::math;


Mesh *
Mesh::createUnregisteredMesh()
{
	return new Mesh();
}


Mesh::Mesh(void) :
	m_pVertexData (0),
	m_IndexData(0),
	m_vMaterialGroups (),
	m_DrawPrimitive(nau::render::IRenderable::TRIANGLES),
	m_Name (""),
	m_NumberOfPrimitives(-1)
{
	m_RealDrawPrimitive = RENDERER->translateDrawingPrimitive(m_DrawPrimitive);
}


Mesh::~Mesh(void)
{
	if (0 != m_pVertexData) 
		delete m_pVertexData;

	if (0 != m_IndexData)
		delete m_IndexData;


	std::vector<nau::material::IMaterialGroup*>::iterator matIter;

	matIter = m_vMaterialGroups.begin();
	
	while (!m_vMaterialGroups.empty()){
		delete((*m_vMaterialGroups.begin()));
		m_vMaterialGroups.erase(m_vMaterialGroups.begin());
	}
}


void 
Mesh::setName (std::string name)
{
	m_Name = name;
}


std::string& 
Mesh::getName (void)
{
	return m_Name;
}


unsigned int
Mesh::getDrawingPrimitive() 
{
	return(m_DrawPrimitive);
}


unsigned int
Mesh::getRealDrawingPrimitive() 
{
	return(m_RealDrawPrimitive);
}


void
Mesh::setDrawingPrimitive(unsigned int aDrawingPrimitive)
{
	m_DrawPrimitive = aDrawingPrimitive;
	m_RealDrawPrimitive = RENDERER->translateDrawingPrimitive(m_DrawPrimitive);
}


void 
Mesh::setNumberOfVerticesPerPatch(int i) 
{
	m_VerticesPerPatch = i;
}


int
Mesh::getnumberOfVerticesPerPatch() 
{
	return m_VerticesPerPatch;

}


nau::render::VertexData& 
Mesh::getVertexData (void)
{
	if (0 == m_pVertexData) {
		m_pVertexData = VertexData::create();
	}
	return (*m_pVertexData);
}


nau::render::IndexData&
Mesh::getIndexData()
{
	createUnifiedIndexVector();
	m_IndexData->setIndexData(&m_UnifiedIndex);
	return *m_IndexData;
}


std::vector<nau::material::IMaterialGroup*>&
Mesh::getMaterialGroups (void)
{
	return (m_vMaterialGroups);
}


void 
Mesh::getMaterialNames(std::set<std::string> *nameList) {

	assert(nameList != 0);

	std::vector<nau::material::IMaterialGroup*>::iterator iter;

	iter = m_vMaterialGroups.begin();

	for ( ; iter != m_vMaterialGroups.end(); ++iter) {

		nameList->insert((*iter)->getMaterialName());
	}
}


void 
Mesh::prepareTriangleIDs(unsigned int sceneObjectID) {

	if (m_DrawPrimitive != IRenderable::TRIANGLES)
		return;

	if (sceneObjectID != 0) {
	
		prepareIndexData();
		createUnifiedIndexVector();

		unsigned int size = m_pVertexData->getDataOf(VertexData::getAttribIndex("position")).size();
		std::vector<VertexData::Attr>* idsArray = new std::vector<VertexData::Attr>(size);

		int primitiveOffset = 3;//getPrimitiveOffset();
		for (unsigned int i = 0; i < size; i++) {
			idsArray->at(i).x = sceneObjectID;
			idsArray->at(i).y = i / primitiveOffset;
		}
		m_pVertexData->setAttributeDataFor(VertexData::getAttribIndex("triangleID"), idsArray);		
	}
}


void 
Mesh::prepareIndexData() 
{
	unsigned int size = m_pVertexData->getDataOf(VertexData::getAttribIndex("position")).size();
	std::vector<int> idsArray = std::vector<int>(size, -1.0f);
	std::vector<int> outlaws;

	createUnifiedIndexVector();

	unsigned int index0, index1, index2, aux0, aux1, aux2;
	for (int i = 0 ; i < getNumberOfVertices()/3; i++) {
	
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
			m_pVertexData->appendVertex(index2);
			m_UnifiedIndex[i * 3 + 2] = size;
			size++;
		}
	}

	// Copy back from UnifiedIndex to MaterialGroups.index
	std::vector<nau::material::IMaterialGroup*>::iterator iter;

	iter = m_vMaterialGroups.begin();
	unsigned int base = 0;
	std::vector<unsigned int>::iterator indexIter;
	indexIter = m_UnifiedIndex.begin();
	for ( ; iter != m_vMaterialGroups.end(); iter ++) {

		unsigned int size = (*iter)->getIndexData().getIndexData().size();
		std::vector<unsigned int>* matGroupIndexes = new std::vector<unsigned int>(size);
		matGroupIndexes->assign(indexIter+base, indexIter+(base+size));
		(*iter)->getIndexData().setIndexData(matGroupIndexes);
		base += size;
	}
}


int 
Mesh::getNumberOfVertices (void)
{
	return (int)(getVertexData().getDataOf (VertexData::getAttribIndex("position"))).size();
}


void 
Mesh::createUnifiedIndexVector() 
{
	m_UnifiedIndex.clear();

	std::vector<nau::material::IMaterialGroup*>::iterator iter;

	iter = m_vMaterialGroups.begin();
	for ( ; iter != m_vMaterialGroups.end(); iter ++) {

		std::vector<unsigned int> matGroupIndexes = (*iter)->getIndexData().getIndexData();
		m_UnifiedIndex.insert(m_UnifiedIndex.end(), matGroupIndexes.begin(),matGroupIndexes.end());
	}
}


void 
Mesh::addMaterialGroup (IMaterialGroup* materialGroup, int offset)
{
	/*
	- search material in vector
	- if it doesn't exist push back
	- if exists merge them
	*/
	std::vector<IMaterialGroup*>::iterator matGroupIter;

	matGroupIter = m_vMaterialGroups.begin();

	for ( ; matGroupIter != m_vMaterialGroups.end(); matGroupIter++ ) {
		IMaterialGroup* aMaterialGroup = (*matGroupIter);

		if (aMaterialGroup->getMaterialName() == materialGroup->getMaterialName()){
			IndexData &indexVertexData = aMaterialGroup->getIndexData();

			indexVertexData.add (materialGroup->getIndexData());
			break;
		}
	}
	if (m_vMaterialGroups.end() == matGroupIter) {
		MaterialGroup *newMat = new MaterialGroup();

		newMat->setMaterialName (materialGroup->getMaterialName());
		newMat->setParent (this);	
		newMat->getIndexData().add (materialGroup->getIndexData());
		m_vMaterialGroups.push_back (newMat);		
	}
}


void 
Mesh::addMaterialGroup (IMaterialGroup* materialGroup, IRenderable *aRenderable)
{
	/* In this case it is necessary to copy the vertices from the 
	 * IRenderable into the local buffer and reindex the materialgroup
	 */
	VertexData &renderableVertexData = aRenderable->getVertexData();

	VertexData *newData = VertexData::create(); 

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
Mesh::merge (nau::render::IRenderable *aRenderable)
{
	VertexData &vVertexData = aRenderable->getVertexData();

	int ofs = getVertexData().add (vVertexData);

	std::vector<IMaterialGroup*> &materialGroups = aRenderable->getMaterialGroups();
	std::vector<IMaterialGroup*>::iterator materialIter;

	materialIter = materialGroups.begin();

	for ( ; materialIter != materialGroups.end(); materialIter++) {
		IMaterialGroup *aMaterialGroup = (*materialIter);
		IndexData &indexData = aMaterialGroup->getIndexData();
		indexData.offsetIndices (ofs);

		addMaterialGroup (aMaterialGroup);
	}
}


std::string 
Mesh::getType (void)
{
	return "Mesh";
}


void 
Mesh::unitize(float min, float max) {

	m_pVertexData->unitize(min,max);
}


void
Mesh::resetCompilationFlags() {

	m_pVertexData->resetCompilationFlag();

	for (unsigned int i = 0; i < m_vMaterialGroups.size(); ++i) {

		m_vMaterialGroups[i]->getIndexData().resetCompilationFlag();
	}
}

//int 
//Mesh::getPrimitiveOffset(void) {
//
//	switch(m_DrawPrimitive) {
//	
//		case nau::render::IRenderer::TRIANGLES: 
//			return (3);
//		case nau::render::IRenderer::TRIANGLE_STRIP: 
//		case nau::render::IRenderer::TRIANGLE_FAN:
//			return (1);
//		case nau::render::IRenderer::LINES:
//			return (2);
//		default: 
//			return (3);
//	}
//}


//int 
//Mesh::getNumberOfPrimitives(void) 
//{
//	switch(m_DrawPrimitive) {
//	
//		case nau::render::IRenderer::TRIANGLES: 
//			return (getNumberOfVertices() / 3);
//		case nau::render::IRenderer::TRIANGLE_STRIP: 
//		case nau::render::IRenderer::TRIANGLE_FAN:
//			return (getNumberOfVertices() - 2);
//		case nau::render::IRenderer::LINES:
//			return (getNumberOfVertices() / 2);
//		default: 
//			return (getNumberOfVertices() / 3);
//	}
//}

//void 
//Mesh::addMaterialGroup (IMaterialGroup* materialGroup, IRenderable *aRenderable)
//{
//	/* In this case it is necessary to copy the vertices from the 
//	 * IRenderable into the local buffer and reindex the materialgroup
//	 */
//	VertexData &renderableVertexData = aRenderable->getVertexData();
//
//	VertexData *newData = VertexData::create(); 
//
//	std::vector<VertexData::Attr> *verticesList = new std::vector<VertexData::Attr>;
//	std::vector<VertexData::Attr> *normalsList = new std::vector<VertexData::Attr>;
//	std::vector<VertexData::Attr> *texCoordsList = new std::vector<VertexData::Attr>;
//	std::vector<VertexData::Attr> *tangentCoordList = new std::vector<VertexData::Attr>;
//	std::vector<VertexData::Attr> *bitangentCoordList = new std::vector<VertexData::Attr>;
//	std::map<unsigned int, unsigned int> newIndicesMap;
//
//
//	std::vector<unsigned int>& indexesList	= materialGroup->getIndexData().getIndexData();
//	std::vector<unsigned int>::iterator indexesIter;
//	
//	indexesIter = indexesList.begin();
//
//	/***MARK***/ //This needs refactoring! Diferent texture coordinates can exist and/or custum attributes
//
//	std::vector<vec4>& verticesPool = renderableVertexData.getDataOf (VertexData::getAttribIndex("position"));
//	std::vector<vec4>& normalsPool = renderableVertexData.getDataOf (VertexData::getAttribIndex("normal")); 
//	std::vector<vec4>& texCoordsPool = renderableVertexData.getDataOf (VertexData::getAttribIndex("texCoord0")); 
//
//	std::vector<vec4>& tangentPool = renderableVertexData.getDataOf (VertexData::getAttribIndex("tangent"));
//	std::vector<vec4>& bitangentPool = renderableVertexData.getDataOf (VertexData::getAttribIndex("binormal"));
//
//	for ( ; indexesIter != indexesList.end(); indexesIter++){
//		if (0 == newIndicesMap.count ((*indexesIter))){
//			verticesList->push_back (verticesPool.at ((*indexesIter)));
//			normalsList->push_back (normalsPool.at ((*indexesIter)));
//			texCoordsList->push_back (texCoordsPool.at ((*indexesIter)));
//			if (tangentPool.size())
//				tangentCoordList->push_back (tangentPool.at ((*indexesIter)));
//			if (bitangentPool.size())
//				bitangentCoordList->push_back (bitangentPool.at ((*indexesIter)));
//
//			newIndicesMap[(*indexesIter)] = (unsigned int)(verticesList->size() - 1);
//			//(*indexesIter) = verticesList->size() - 1;
//		} //else {
//		(*indexesIter) = newIndicesMap[(*indexesIter)];
//		//}
//	}
//
//	newData->setDataFor (VertexData::getAttribIndex("position"), verticesList);
//	newData->setDataFor (VertexData::getAttribIndex("normal"), normalsList);
//	newData->setDataFor (VertexData::getAttribIndex("texCoord0"), texCoordsList);
//	newData->setAttributeDataFor (VertexData::getAttribIndex("tangent"), tangentCoordList, VertexData::NOLOC);
//	newData->setAttributeDataFor (VertexData::getAttribIndex("binormal"), bitangentCoordList, VertexData::NOLOC);
//
//	int offset = getVertexData().add (*newData);
//	delete newData;
//	
//	materialGroup->getIndexData().offsetIndices (offset);
//	addMaterialGroup (materialGroup);			
//}