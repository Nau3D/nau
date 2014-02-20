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
	m_IndexSize (0)
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

unsigned int
IndexData::getIndexSize (void) 
{
	return m_IndexSize;
}

void
IndexData::setIndexData (std::vector<unsigned int>* indexData)
{
	if (m_InternalIndexArray != 0)
		delete m_InternalIndexArray;

	m_InternalIndexArray = indexData;
	m_IndexSize = (unsigned int)(indexData->size());
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

		} else {

			aIndexVec.insert (aIndexVec.end(), indexData.begin(), indexData.end());
			m_IndexSize = (unsigned int)(aIndexVec.size());
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

