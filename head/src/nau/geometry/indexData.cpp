#include "nau/geometry/indexData.h"

#include "nau/config.h"
#ifdef NAU_OPENGL
#include "nau/render/opengl/glIndexArray.h"
#endif

#include <assert.h>


using namespace nau::geometry;
using namespace nau::render;
using namespace nau::math;


std::shared_ptr<IndexData>
IndexData::Create(std::string &name) {

#ifdef NAU_OPENGL
	return std::shared_ptr<IndexData>(new GLIndexArray(name));
#elif NAU_DIRECTX
	i = std::shared_ptr<IndexData>(new DXIndexArray(name));
#endif
}


IndexData::IndexData(void) :
	m_UseAdjacency(false) {

}


IndexData::IndexData(std::string & name):
	m_UseAdjacency(false),
	m_Name(name) {

}


IndexData::~IndexData(void) {

}


void 
IndexData::setName(std::string name) {

	m_Name = name;
}


std::shared_ptr<std::vector<unsigned int>> &
IndexData::getIndexData (void) {

	return m_InternalIndexArray;
}


#ifdef NAU_OPTIX
void
IndexData::getIndexDataAsInt(std::vector<int> *v) {

	if (!m_InternalIndexArray)
		return;
	else {
		v->resize(m_InternalIndexArray->size());
		for (unsigned int i = 0; i < m_InternalIndexArray->size(); ++i) {

			v->at(i) = (int)m_InternalIndexArray->at(i);
		}
	}
}
#endif


unsigned int
IndexData::getIndexSize (void) {

	if (m_InternalIndexArray && m_UseAdjacency == false)
		return (unsigned int)m_InternalIndexArray->size();
	else if (m_AdjIndexArray)
		return (unsigned int)m_AdjIndexArray->size();
	else
		return 0;
}


void
IndexData::setIndexData (std::shared_ptr<std::vector<unsigned int>> &indexData) {

	if (m_InternalIndexArray && m_InternalIndexArray != indexData) {
		m_InternalIndexArray.reset();
	}

	m_InternalIndexArray = indexData;
}


void
IndexData::add (std::shared_ptr<IndexData> &aIndexData) {

	size_t offset = 0;

	std::shared_ptr<std::vector<unsigned int>> &indexData = aIndexData->getIndexData();

	if (indexData) {
		std::shared_ptr<std::vector<unsigned int>> &aIndexVec = getIndexData();

		if (!aIndexVec) {
			aIndexVec.reset(new std::vector<unsigned int>);
				//std::shared_ptr<std::vector<unsigned int>>(>);

			aIndexVec->insert (aIndexVec->begin(), indexData->begin(), indexData->end());
			m_UseAdjacency = aIndexData->getAdjacency();

		} 
		else {
			aIndexVec->insert (aIndexVec->end(), indexData->begin(), indexData->end());
		}
	}
}


void
IndexData::offsetIndices (int amount) {

	if (!m_InternalIndexArray)
		return;

	std::vector<unsigned int>::iterator indIter;

	indIter = m_InternalIndexArray->begin();

	for (; indIter != m_InternalIndexArray->end(); indIter++) {
		(*indIter) += amount;
	}
}


void 
IndexData::buildAdjacencyList() {
	
	size_t indC = m_InternalIndexArray->size();
	unsigned int *faceArray = (unsigned int *)&(m_InternalIndexArray->at(0));
	size_t triC =  indC / 3;

	// Create the half edge structure
	std::map<std::pair<unsigned int,unsigned int>, struct HalfEdge *> myEdges;
	struct HalfEdge *edge;

	// fill it up with edges. twin info will be added latter
	edge = (struct HalfEdge *)malloc(sizeof(struct HalfEdge) * triC * 3);
	for (unsigned int i = 0; i < triC; ++i) {
		
		edge[i*3].vertex = faceArray[i*3+1];
		edge[i*3+1].vertex = faceArray[i*3+2];
		edge[i*3+2].vertex = faceArray[i*3];

		edge[i*3].next = &edge[i*3+1];
		edge[i*3+1].next = &edge[i*3+2];
		edge[i*3+2].next = &edge[i*3];

		edge[i*3].twin   = NULL;
		edge[i*3+1].twin = NULL;
		edge[i*3+2].twin = NULL;

		myEdges[std::pair<unsigned int,unsigned int>(faceArray[i*3+2],faceArray[i*3])] = &edge[i*3];
		myEdges[std::pair<unsigned int,unsigned int>(faceArray[i*3],faceArray[i*3+1])] = &edge[i*3+1];
		myEdges[std::pair<unsigned int,unsigned int>(faceArray[i*3+1],faceArray[i*3+2])] = &edge[i*3+2];
	}

	// add twin info
	std::map<std::pair<unsigned int,unsigned int>, struct HalfEdge *>::iterator iter;
	std::pair<unsigned int,unsigned int> edgeIndex, twinIndex;

	iter = myEdges.begin();

	for (; iter != myEdges.end(); ++iter) {
		 
		edgeIndex = iter->first;
		twinIndex = std::pair<unsigned int, unsigned int>(edgeIndex.second, edgeIndex.first);

		if (myEdges.count(twinIndex))
			iter->second->twin = myEdges[twinIndex];
		else
			iter->second->twin = NULL;
	}
		
	if (m_AdjIndexArray)
		m_AdjIndexArray.reset();
	m_AdjIndexArray = std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>);
	m_AdjIndexArray->resize(triC*6);
	std::vector<unsigned int> &aux = *m_AdjIndexArray.get();
	for (unsigned int i = 0; i < triC; i++) {
		
		// NOTE: twin may be null
		aux[i*6]   = edge[3*i + 0].next->vertex;
		aux[i*6+1] = edge[3*i + 0].twin?edge[3*i + 0].twin->vertex:edge[3*i + 0].next->vertex;

		aux[i*6+2] = edge[3*i + 1].next->vertex;
		aux[i*6+3] = edge[3*i + 1].twin?edge[3*i + 1].twin->vertex:edge[3*i + 1].next->vertex;

		aux[i*6+4] = edge[3*i + 2].next->vertex;
		aux[i*6+5] = edge[3*i + 2].twin?edge[3*i + 2].twin->vertex:edge[3*i + 2].next->vertex;

	}
	free(edge);
}