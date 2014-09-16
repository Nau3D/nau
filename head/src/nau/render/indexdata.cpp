#include <nau/render/indexdata.h>
#include <nau/config.h>
#include <assert.h>

#ifdef NAU_OPENGL
#include <nau/render/opengl/glindexarray.h>
#endif

using namespace nau::render;
using namespace nau::math;

std::vector<unsigned int> IndexData::NoIndexData;



IndexData* 
IndexData::create (void)
{
#ifdef NAU_OPENGL
	return new GLIndexArray;
#elif NAU_DIRECTX
	return new DXIndexArray;
#else
	return 0;
#endif
}


IndexData::IndexData(void) :
	m_InternalIndexArray (0),
	m_AdjIndexArray(0),
	//m_IndexSize (0),
	m_UseAdjacency(false)
{
}


IndexData::~IndexData(void)
{
	if (0 != m_InternalIndexArray) {
		delete m_InternalIndexArray;
	}
}


std::vector<unsigned int>&
IndexData::getIndexData (void)
{
	if (0 == m_InternalIndexArray) {
		return IndexData::NoIndexData;
	}

	return (*m_InternalIndexArray);
}

#ifdef NAU_OPTIX_PRIME
std::vector<int>*
IndexData::getIndexDataAsInt(void) {

	std::vector < int >* v = new std::vector < int >;

	if (NULL == m_InternalIndexArray)
		v->resize(0);
	else {
		v->resize(m_InternalIndexArray->size());
		for (unsigned int i = 0; i < m_InternalIndexArray->size(); ++i) {

			v->at(i) = (int)m_InternalIndexArray->at(i);
		}
	}
	return v;
}
#endif

unsigned int
IndexData::getIndexSize (void) 
{
	if (m_InternalIndexArray != NULL && m_UseAdjacency == false)
		return m_InternalIndexArray->size();
	else 
		return m_AdjIndexArray.size();
	//else
	//	return 0;
}


void
IndexData::setIndexData (std::vector<unsigned int>* indexData)
{
	if (m_InternalIndexArray != 0)
		delete m_InternalIndexArray;

	m_InternalIndexArray = indexData;
}


int
IndexData::add (IndexData &aIndexData)
{
	size_t offset = 0;


	std::vector<unsigned int> &indexData = aIndexData.getIndexData();
	if (IndexData::NoIndexData != indexData) {

		std::vector<unsigned int> &aIndexVec = getIndexData();

		if (aIndexVec == IndexData::NoIndexData) {

			std::vector<unsigned int> *aNewVector = new std::vector<unsigned int>;
			aNewVector->insert (aNewVector->begin(), indexData.begin(), indexData.end());
			setIndexData (aNewVector);
			m_UseAdjacency = aIndexData.getAdjacency();

		} else {

			aIndexVec.insert (aIndexVec.end(), indexData.begin(), indexData.end());
		}
	}

	return (int)offset;
}


void
IndexData::offsetIndices (int amount)
{
	std::vector<unsigned int> &indices = getIndexData();
	std::vector<unsigned int>::iterator indIter;

	indIter = indices.begin();

	for ( ; indIter != indices.end(); indIter++) {
		(*indIter) += amount;
	}
}


void 
IndexData::buildAdjacencyList() {
	
	unsigned int indC = m_InternalIndexArray->size();
	unsigned int *faceArray = (unsigned int *)&(m_InternalIndexArray->at(0));
	unsigned int triC =  indC / 3;

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
		
	m_AdjIndexArray.resize(triC*6);// = (unsigned int *)malloc(sizeof(unsigned int) * mesh->mNumFaces * 6);
	for (unsigned int i = 0; i < triC; i++) {
		
		// NOTE: twin may be null
		m_AdjIndexArray[i*6]   = edge[3*i + 0].next->vertex;
		m_AdjIndexArray[i*6+1] = edge[3*i + 0].twin?edge[3*i + 0].twin->vertex:edge[3*i + 0].next->vertex;

		m_AdjIndexArray[i*6+2] = edge[3*i + 1].next->vertex;
		m_AdjIndexArray[i*6+3] = edge[3*i + 1].twin?edge[3*i + 1].twin->vertex:edge[3*i + 1].next->vertex;

		m_AdjIndexArray[i*6+4] = edge[3*i + 2].next->vertex;
		m_AdjIndexArray[i*6+5] = edge[3*i + 2].twin?edge[3*i + 2].twin->vertex:edge[3*i + 2].next->vertex;

	}	
}