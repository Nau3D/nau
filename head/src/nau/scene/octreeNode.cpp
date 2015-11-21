#include "nau/scene/octreeNode.h"

#include "nau/slogger.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"
#include "nau/math/matrix.h"
#include "nau.h"

#include <assert.h>

using namespace nau::scene;
using namespace nau::render;
using namespace nau::material;
using namespace nau::geometry;
using namespace nau::math;

int OctreeNode::counter = 0;


OctreeNode::OctreeNode () :
	SceneObject(),
	m_pParent (0),
	m_ChildCount (0),
	m_Divided (false),
	m_NodeId (0),
	m_NodeDepth (0),
	m_pLocalMesh (0)
{
	for (int i = 0; i < 8; i++) {
		m_pChilds[i] = 0;
	}

//	m_Transform = new SimpleTransform;
}


OctreeNode::OctreeNode (OctreeNode *parent, IBoundingVolume *aBoundingVolume, int nodeId, int nodeDepth) :
	SceneObject(),
	m_pParent (parent),
	m_ChildCount (0),
	m_Divided (false),
	m_NodeId (nodeId),
	m_NodeDepth (nodeDepth),
	m_pLocalMesh (0)
{
	for (int i = 0; i < 8; i++) {
		m_pChilds[i] = 0;
	}

	//m_Transform = new SimpleTransform;
	setBoundingVolume (aBoundingVolume);
}

OctreeNode::~OctreeNode(void)
{
	for (int i = 0; i < 8; i++) {
		delete m_pChilds[i];
	}
	delete m_pLocalMesh;
	/***MARK***/ //Delete children
}


void
OctreeNode::eventReceived(const std::string &sender,
	const std::string &eventType,
	const std::shared_ptr<IEventData> &evt) {

}


void 
OctreeNode::getMaterialNames(std::set<std::string> *nameList)
{
	if (m_pLocalMesh)
		m_pLocalMesh->getMaterialNames(nameList);

	for (int i = 0; i < 8 ; i++) {

		if (m_pChilds[i])
	
			m_pChilds[i]->getMaterialNames(nameList);
	}
}

//void
//OctreeNode::addRenderable (nau::render::IRenderable *aRenderable)
//{
//	
//	VertexData &vVertexData = aRenderable->getVertexData();
//
//	if (true == m_Divided) {
//		Mesh *tempMesh[9] = { 0 };
//
//		//For each material group
//		std::vector<IMaterialGroup*> &vMaterialGroups =
//			aRenderable->getMaterialGroups();
//		std::vector<IMaterialGroup*>::iterator matIter;
//
//		matIter = vMaterialGroups.begin();
//
//		for ( ; matIter != vMaterialGroups.end(); matIter++) {
//			MaterialGroup *tempMaterialGroup[9] = { 0 };
//
//			IMaterialGroup *pMaterialGroup = (*matIter);
//
//			IndexData &VertexDataMaterialGroup = pMaterialGroup->getIndexData();
//			std::vector<unsigned int> &vIndexData = VertexDataMaterialGroup.getIndexData();
//			std::vector<unsigned int>::iterator indexIter;
//
//			indexIter = vIndexData.begin();
//
//			// For each triangle, index, index + 1, index + 2, check to which octant it belongs to
//			try {
//				for ( ; indexIter != vIndexData.end(); indexIter += 3) {
//					VertexData::Attr &v1 = vVertexData.getDataOf (VertexData::getAttribIndex("position")).at (*(indexIter));
//					VertexData::Attr &v2 = vVertexData.getDataOf (VertexData::getAttribIndex("position")).at (*(indexIter+1));
//					VertexData::Attr &v3 = vVertexData.getDataOf (VertexData::getAttribIndex("position")).at (*(indexIter+2));
//						
//					int v1Octant = _octantFor (v1);
//					int v2Octant = _octantFor (v2);
//					int v3Octant = _octantFor (v3);
//
//					int index = v1Octant;
//
//					if (v1Octant != v2Octant || v2Octant != v3Octant) {
//						index = ROOT;
//					}
//
//					if (0 == tempMaterialGroup[index]) {
//						tempMaterialGroup[index] = new MaterialGroup;
//
//						tempMaterialGroup[index]->setMaterialName (pMaterialGroup->getMaterialName());
//						//tempMaterialGroup[index]->setMaterialId (pMaterialGroup->getMaterialId());
//
//					}
//
//					std::vector<unsigned int> &vTempIndexData = 
//						tempMaterialGroup[index]->getIndexData().getIndexData();
//
//					if (IndexData::NoIndexData == vTempIndexData) {
//						std::vector<unsigned int>* newIndexData = new std::vector<unsigned int>;
//
//						(tempMaterialGroup[index]->getIndexData()).setIndexData (newIndexData);
//						newIndexData->push_back (*(indexIter));
//						newIndexData->push_back (*(indexIter+1));
//						newIndexData->push_back (*(indexIter+2));
//					} else {
//						vTempIndexData.push_back (*(indexIter));
//						vTempIndexData.push_back (*(indexIter+1));
//						vTempIndexData.push_back (*(indexIter+2));
//					}
//				}
//
//			}
//			catch (std::exception& so) {
//				LOG_INFO ("Exception: %s", so.what());	
//				assert (1);
//			}
//
//			for (int index = TOPFRONTLEFT; index <= ROOT; index++) {
//				if (0 != tempMaterialGroup[index]) {
//					if (0 == tempMesh[index]) {
//						tempMesh[index] = (Mesh *)RESOURCEMANAGER->createRenderable("Mesh",_genOctName());//new Mesh;
//						//tempMesh[index]->setName(_genOctName());
//					}
//					tempMesh[index]->addMaterialGroup (tempMaterialGroup[index], aRenderable);
//					delete tempMaterialGroup[index];
//					tempMaterialGroup[index] = 0;
//				}
//			}
//		}
//
//		for (int index = TOPFRONTLEFT; index <= BOTTOMBACKRIGHT; index++) {
//			if (0 != tempMesh[index]) {
//				if (0 == m_pChilds[index]) {
//					m_pChilds[index] = _createChild (index);
//				}
//				m_pChilds[index]->setRenderable (tempMesh[index]);
//				delete tempMesh[index];
//				tempMesh[index] = 0;
//			}
//		}
//
//		if (0 != tempMesh[ROOT]) {
//			if (0 == m_pLocalMesh) {
//				m_pLocalMesh = (Mesh *)RESOURCEMANAGER->createRenderable("Mesh", _genOctName());//new Mesh;
//			//	m_pLocalMesh->setName ();
//				m_Renderable = m_pLocalMesh;
//			}
//			m_pLocalMesh->merge (tempMesh[ROOT]);
//			delete tempMesh[ROOT];
//			tempMesh[ROOT] = 0;
//		}
//		
//	} else { //It is not divided
//		if ((0 != m_pLocalMesh) && (m_pLocalMesh->getNumberOfVertices() / 3) > OctreeNode::MAXPRIMITIVES)
//		{
//			m_Divided = true;
//
//			Mesh *pTempMesh = m_pLocalMesh;
//			
//			m_pLocalMesh = 0;
//
//			// Inherited by SceneObject
//			m_Renderable = 0;
//
//			addRenderable (pTempMesh);
//			delete pTempMesh;
//
//			addRenderable (aRenderable);
//			
//		} else {
//			
//			if (0 == m_pLocalMesh) {
//				m_pLocalMesh = (Mesh *)RESOURCEMANAGER->createRenderable("Mesh", _genOctName()); //new Mesh;
//				//m_pLocalMesh->setName ();
//				m_Renderable = m_pLocalMesh;
//
//			}
//			m_pLocalMesh->merge (aRenderable);
//
//		}
//	}
////	tightBoundingVolume();
//}


void OctreeNode::tightBoundingVolume() {

	if (0 != m_BoundingVolume) {
		delete m_BoundingVolume;
	}

	m_BoundingVolume = new BoundingBox; /***MARK***/
	if (0 != m_pLocalMesh)
		m_BoundingVolume->calculate (m_pLocalMesh->getVertexData()->getDataOf (VertexData::GetAttribIndex(std::string("position"))));

	for (int i = TOPFRONTLEFT; i < ROOT; i++) {
	
		if (0 != m_pChilds[i]) {
			m_BoundingVolume->compound((m_pChilds[i]->m_BoundingVolume));
		}
	}

}




void 
OctreeNode::updateNodeTransform(nau::math::mat4 &t)
{
	if (0 != m_pLocalMesh) {
		updateGlobalTransform(t);
	}
	for (int i = TOPFRONTLEFT; i < ROOT; i++) {
	
		if (0 != m_pChilds[i]) {
			m_pChilds[i]->updateNodeTransform(t);
		}
	}

}


void OctreeNode::unitize(vec3 &center, vec3 &min, vec3 &max) {
	
	m_pLocalMesh->unitize(center, min, max);

	for (int i = OctreeNode::TOPFRONTLEFT; i < OctreeNode::ROOT;  i++) {
	
		if (0 != m_pChilds[i]) {
			m_pChilds[i]->unitize(center, min, max);
		}
	}
	tightBoundingVolume();
}

void 
OctreeNode::setRenderable (nau::render::IRenderable *aRenderable)
{
	int vertexArrayPos = VertexData::GetAttribIndex(std::string("position"));
	int offSet;

	if (m_pLocalMesh) {
		delete m_pLocalMesh;
		m_pLocalMesh = 0;
	}

	m_pLocalMesh = (Mesh *)aRenderable;//dynamic_cast<Mesh*>(aRenderable);
	m_Renderable = aRenderable;

	if (m_pLocalMesh->getNumberOfVertices()/3 > MAXPRIMITIVES) {
	
		std::shared_ptr<VertexData> &vVertexData = aRenderable->getVertexData();
		Mesh *tempMesh[9] = { 0 };

		//Octree are only implemented for triangles
		offSet = 3; // m_pLocalMesh->getPrimitiveOffset();

		//For each material group
		std::vector<std::shared_ptr<MaterialGroup>> &vMaterialGroups =
			aRenderable->getMaterialGroups();

		for (auto& pMaterialGroup: vMaterialGroups) {
			std::shared_ptr<MaterialGroup> tempMaterialGroup[9];

			std::shared_ptr<IndexData> &VertexDataMaterialGroup = pMaterialGroup->getIndexData();
			std::shared_ptr<std::vector<unsigned int>> &vIndexData = VertexDataMaterialGroup->getIndexData();
			std::vector<unsigned int>::iterator indexIter;

			indexIter = vIndexData->begin();
			std::shared_ptr<std::vector<VertexData::Attr>> &vVertices = vVertexData->getDataOf(vertexArrayPos);

			// For each triangle, index, index + 1, index + 2, check to which octant it belongs to
			try {
				for (unsigned int i = 0; i < vIndexData->size(); i += offSet) {
					// carefull: three vertices are only for triangles, not lines
					// strips and fans have their own set of rules for indexes
					VertexData::Attr &v1 = vVertices->at (vIndexData->at(i));
					VertexData::Attr &v2 = vVertices->at (vIndexData->at(i+1));
					VertexData::Attr &v3 = vVertices->at (vIndexData->at(i+2));
						
					int v1Octant = _octantFor (v1);
					int v2Octant = _octantFor (v2);
					int v3Octant = _octantFor (v3);

					int index = v1Octant;

					if (v1Octant != v2Octant || v2Octant != v3Octant) {
						index = ROOT;
					}

					if (0 == tempMaterialGroup[index]) {
						tempMaterialGroup[index] = MaterialGroup::Create(NULL, pMaterialGroup->getMaterialName());

						//tempMaterialGroup[index]->setMaterialName (pMaterialGroup->getMaterialName());
						//tempMaterialGroup[index]->setMaterialId (pMaterialGroup->getMaterialId());
					}

					std::shared_ptr<std::vector<unsigned int>> &vTempIndexData =
						tempMaterialGroup[index]->getIndexData()->getIndexData();

					if (vTempIndexData) {
						std::shared_ptr<std::vector<unsigned int>> newIndexData = 
							std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>);
						// carefull: order of triangles is relevant for strips
						(tempMaterialGroup[index]->getIndexData())->setIndexData (newIndexData);
						newIndexData->push_back (vIndexData->at(i));
						newIndexData->push_back (vIndexData->at(i+1));
						newIndexData->push_back (vIndexData->at(i+2));
					} else {
						vTempIndexData->push_back (vIndexData->at(i));
						vTempIndexData->push_back (vIndexData->at(i+1));
						vTempIndexData->push_back (vIndexData->at(i+2));
					}
				}

			}
			catch (std::exception& so) {
				LOG_INFO ("Exception: %s", so.what());	
				assert (1);
			}

			for (int index = TOPFRONTLEFT; index <= ROOT; index++) {
				if (0 != tempMaterialGroup[index]) {
					if (0 == tempMesh[index]) {
						tempMesh[index] = (Mesh *)RESOURCEMANAGER->createRenderable("Mesh", _genOctName()); //new Mesh;
						//tempMesh[index]->setName(_genOctName());
					}
					tempMesh[index]->addMaterialGroup (tempMaterialGroup[index], aRenderable);
					tempMaterialGroup[index].reset();
				}
			}
		}

 		for (int index = TOPFRONTLEFT; index <= BOTTOMBACKRIGHT; index++) {
			if (0 != tempMesh[index]) {
				if (0 == m_pChilds[index]) {
					m_pChilds[index] = _createChild (index);
				}
				m_pChilds[index]->setRenderable (tempMesh[index]);
				//delete tempMesh[index];
				//tempMesh[index] = 0;
			}
		}

		if (m_pLocalMesh) {
			delete m_pLocalMesh;
			m_pLocalMesh = 0;
		}

		if (0 != tempMesh[ROOT]) {		
			m_pLocalMesh = tempMesh[ROOT];
			m_pLocalMesh->setName (_genOctName());
			m_Renderable = m_pLocalMesh;
			//delete tempMesh[ROOT];
			//tempMesh[ROOT] = 0;
		}

		// split
	}
}


std::string 
OctreeNode::getType (void)
{
	return "OctreeNode";
}

int
OctreeNode::_octantFor (VertexAttrib& v)
{
	int octant = 8;

	const vec3 &BBCenter = m_BoundingVolume->getCenter();

	if (v.x > BBCenter.x)	{
		if (v.y > BBCenter.y)	{
			if (v.z > BBCenter.z)	{
				octant = TOPFRONTRIGHT;
			} else {
				octant = TOPBACKRIGHT;
			}
		} else {
			if (v.z > BBCenter.z) {
				octant = BOTTOMFRONTRIGHT;
			} else {
				octant = BOTTOMBACKRIGHT;
			}
		}
	} else {
		if (v.y > BBCenter.y) {
			if (v.z > BBCenter.z) {
				octant = TOPFRONTLEFT;				
			} else {
				octant = TOPBACKLEFT;
			}
		} else {
			if (v.z > BBCenter.z) {
				octant = BOTTOMFRONTLEFT;
			} else {
				octant = BOTTOMBACKLEFT;
			}
		}
	}

	return octant;
}

OctreeNode*
OctreeNode::_createChild (int octant)
{
	vec3 bbMin;
	vec3 bbMax;

	const vec3 &LocalBBMin = m_BoundingVolume->getMin();
	const vec3 &LocalBBMax = m_BoundingVolume->getMax();
	const vec3 &LocalBBCenter = m_BoundingVolume->getCenter();

	switch (octant) {
		case OctreeNode::TOPFRONTLEFT: {
			bbMin.x = LocalBBMin.x;
			bbMin.y = LocalBBCenter.y;
			bbMin.z = LocalBBCenter.z;
			
			bbMax.x = LocalBBCenter.x;
			bbMax.y = LocalBBMax.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeNode::TOPFRONTRIGHT: {
			bbMin.x = LocalBBCenter.x;
			bbMin.y = LocalBBCenter.y;
			bbMin.z = LocalBBCenter.z;
			
			bbMax.x = LocalBBMax.x;
			bbMax.y = LocalBBMax.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeNode::TOPBACKLEFT: {
			bbMin.x = LocalBBMin.x;
			bbMin.y = LocalBBCenter.y;
			bbMin.z = LocalBBMin.z;
			
			bbMax.x = LocalBBCenter.x;
			bbMax.y = LocalBBMax.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeNode::TOPBACKRIGHT: {
			bbMin.x = LocalBBCenter.x;
			bbMin.y = LocalBBCenter.y;
			bbMin.z = LocalBBMin.z;
			
			bbMax.x = LocalBBMax.x;
			bbMax.y = LocalBBMax.y;
			bbMax.z = LocalBBCenter.z;
			break;
		}
		case OctreeNode::BOTTOMFRONTLEFT: {
			bbMin.x = LocalBBMin.x; 
			bbMin.y = LocalBBMin.y;
			bbMin.z = LocalBBCenter.z;
			
			bbMax.x = LocalBBCenter.x;
			bbMax.y = LocalBBCenter.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeNode::BOTTOMFRONTRIGHT: {
			bbMin.x = LocalBBCenter.x;
			bbMin.y = LocalBBMin.y;
			bbMin.z = LocalBBCenter.z;
			
			bbMax.x = LocalBBMax.x;
			bbMax.y = LocalBBCenter.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeNode::BOTTOMBACKLEFT: {
			bbMin.x = LocalBBMin.x; 
			bbMin.y = LocalBBMin.y;
			bbMin.z = LocalBBMin.z;
			
			bbMax.x = LocalBBCenter.x;
			bbMax.y = LocalBBCenter.y;
			bbMax.z = LocalBBCenter.z;
			break;
		}
		case OctreeNode::BOTTOMBACKRIGHT: {
			bbMin.x = LocalBBCenter.x;
			bbMin.y = LocalBBMin.y;
			bbMin.z = LocalBBMin.z;

			bbMax.x = LocalBBMax.x;
			bbMax.y = LocalBBCenter.y;
			bbMax.z = LocalBBCenter.z;
			break;
		}
	}
	
	this->m_ChildCount++;
	return new OctreeNode (this, new BoundingBox (bbMin, bbMax), octant, m_NodeDepth + 1);
}

void
OctreeNode::_compile (void)
{
	if (0 != m_pLocalMesh) {
		m_pLocalMesh->getVertexData()->compile();
		std::vector<std::shared_ptr<MaterialGroup>> &matGroups = m_pLocalMesh->getMaterialGroups();

		for (auto& matGroupsIter: matGroups){
			matGroupsIter->compile();
		}

	}

	for (int i = 0; i < 8; i++) {
		if (0 != m_pChilds[i]) {
			m_pChilds[i]->_compile();
		}
	}
}

void 
OctreeNode::_findVisibleSceneObjects (std::vector<SceneObject*> &m_vReturnVector,
																Frustum &aFrustum, 
																Camera &aCamera,
																bool conservative)
{
//	if (m_pLocalMesh == 0)
//		return;

	int side = aFrustum.isVolumeInside (getBoundingVolume(), conservative);

	//if (Frustum::OUTSIDE == side) {
	//	return;
	//}

	if (0 != m_pLocalMesh) {
		m_vReturnVector.push_back (this);
	}

	for (int i = TOPFRONTLEFT; i <= BOTTOMBACKRIGHT; i++) {
		if (0 != m_pChilds[i]) {
			m_pChilds[i]->_findVisibleSceneObjects (m_vReturnVector, aFrustum, aCamera, conservative);	
		}
	}
}

OctreeNode*
OctreeNode::_getChild (int i) 
{
	return m_pChilds[i];
}

void 
OctreeNode::_setChild (int i, OctreeNode *aNode)
{
	m_pChilds[i] = aNode;
}


void
OctreeNode::_setParent (OctreeNode *parent)
{
	m_pParent = parent;
}

int
OctreeNode::_getChildCount (void)
{
	return m_ChildCount;
}

void 
OctreeNode::writeSpecificData (std::fstream &f)
{
	f.write ((char *)&m_ChildCount, sizeof (m_ChildCount));
	f.write ((char *)&m_NodeId, sizeof (m_NodeId));
	f.write ((char *)&m_NodeDepth, sizeof (m_NodeDepth));
}

void 
OctreeNode::readSpecificData (std::fstream &f)
{
	f.read ((char *)&m_ChildCount, sizeof (m_ChildCount));
	f.read ((char*)&m_NodeId, sizeof (int));
	f.read ((char*)&m_NodeDepth, sizeof (int));
}

std::string
OctreeNode::_genOctName (void)
{
	char name[256];
	int r = rand();
	counter++;
	sprintf (name, "oct%d", counter);
	return name;
}


void 
OctreeNode::resetCounter() {

	counter = 0;
}
/*

OctreeNode::OctreeNode(Octree *master, vec3 bbMin, vec3 bbMax, int nodeId, int nodeDepth):
	m_Master (master), 
	m_ChildCount (0),
	m_Divided (false),
	m_TriangleCount (0), 
	m_bbMin (bbMin), 
	m_bbMax (bbMax), 
	m_bbCenter(bbMax), 
	m_MaterialGroups (),
	m_NodeId (nodeId),
	m_NodeDepth (nodeDepth)
{
	for (int i = 0; i < 8; i++){
		m_Childs[i] = 0;
	}

	m_bbCenter += m_bbMin;

	m_bbCenter *= 0.5;	
	
	//LOG_INFO("Bounding Box Max: (%.4f, %.4f, %.4f)", m_bbMax.x, m_bbMax.y, m_bbMax.z);
	//LOG_INFO("Bounding Box Min: (%.4f, %.4f, %.4f)", m_bbMin.x, m_bbMin.y, m_bbMin.z);
	//LOG_INFO("Bounding Box center: (%.4f, %.4f, %.4f)", m_bbCenter.x, m_bbCenter.y, m_bbCenter.z);
}

void 
OctreeNode::addMaterialGroup (IMaterialGroup& aMaterialGroup)
{

	//int triangleCount = indexList.size () / 3;

	LOG_INFO ("(%d,%d)Triangle count: %d", m_NodeId, m_NodeDepth, m_TriangleCount);
	//LOG_INFO ("Adding material: %s", (aMaterialGroup.getMaterialName()).c_str());

	if (true == m_Divided){
		MaterialGroup *temporaryMaterialGroups[9];

		std::vector<unsigned int> indexList = aMaterialGroup.getIndexList();

		for (int i = 0; i < 9; i++){
			temporaryMaterialGroups[i] = 0; 
		}

		//Check for each group of 3 vertices were they belong
		//If the vertices belong to diferent octants, then the 3 remain in this node.
		std::vector<unsigned int>::iterator indexListIter;
		for (indexListIter = indexList.begin ();
			indexListIter != indexList.end ();
			indexListIter += 3){
				

				vec3& v1 = m_Master->getVertice (*indexListIter);
				//LOG_INFO ("V1@%d: %.2f, %.2f, %.2f", *indexListIter, v1.x, v1.y, v1.z);

				vec3& v2 = m_Master->getVertice (*(indexListIter+1));
				//LOG_INFO ("V2@%d: %.2f, %.2f, %.2f", *(indexListIter+1), v2.x, v2.y, v2.z);

				vec3& v3 = m_Master->getVertice (*(indexListIter+2));
				//LOG_INFO ("V3@%d: %.2f, %.2f, %.2f", *(indexListIter+2), v3.x, v3.y, v3.z);

				int v1Oct = octantFor (v1);
				int v2Oct = octantFor (v2);
				int v3Oct = octantFor (v3);
				
				int index = v1Oct;
				//Check if all the vertices belong to the same octant

				if (v1Oct != v2Oct || v2Oct != v3Oct){
					index = 8;
				} else {
					//PHONY
				}

				if (0 == temporaryMaterialGroups[index]){
					temporaryMaterialGroups[index] = new MaterialGroup;

					temporaryMaterialGroups[index]->setMaterialId (aMaterialGroup.getMaterialId ());
					temporaryMaterialGroups[index]->setMaterialName (aMaterialGroup.getMaterialName ());
				} else {
					//PHONY
				}


				std::vector<unsigned int>& indexList = temporaryMaterialGroups[index]->getIndexList ();
				indexList.push_back (*indexListIter);
				indexList.push_back (*(indexListIter+1));
				indexList.push_back (*(indexListIter+2));
		}
		
		//Now that we know which vertices belong to each octant, it's time to fill the octants
		
		//First let us deal with those that remain in this node.
		if (0 != temporaryMaterialGroups[8]){
			this->appendMaterialGroup (*temporaryMaterialGroups[8]);
		} else {
			//PHONY
		}

		for (int i = 0; i < 8; i++){
			if (0 != temporaryMaterialGroups[i]){
				if (0 == m_Childs[i]){
					m_Childs[i] = createNewOctantChild(i);
					m_ChildCount++;
				} else {
					//PHONY
				}
				m_Childs[i]->addMaterialGroup (*(temporaryMaterialGroups[i]));
			}
		}

		for (int i = 0; i < 9; i++){
			if (0 != temporaryMaterialGroups[i]){
				delete temporaryMaterialGroups[i];
			} else {
				//PHONY
			}
		}	
	} else { // If it is not already divided...
		if (this->m_TriangleCount > OctreeNode::MAXTRIANGLES){
			m_Divided = true;

			LOG_INFO ("(%d,%d)===Node division===", m_NodeId, m_NodeDepth);

			std::vector<MaterialGroup*> copyOfMaterialGroups(m_MaterialGroups);
			std::vector<MaterialGroup*>::iterator materialGroupIter = copyOfMaterialGroups.begin();
			

			for (unsigned int i = 0; i < m_MaterialGroups.size(); i++){
				m_MaterialGroups[i] = 0;
			}

			m_MaterialGroups.clear();
			m_TriangleCount = 0;
			
			for(; materialGroupIter != copyOfMaterialGroups.end();
				materialGroupIter++){
					this->addMaterialGroup (**materialGroupIter);
					delete (*materialGroupIter);
			}

			this->addMaterialGroup (aMaterialGroup);
		} else { //Guess we don't need to divide it
			this->appendMaterialGroup (aMaterialGroup);
		}
	}
}

bool 
OctreeNode::hasChildren()
{
	return (0 != this->m_ChildCount);
}


std::vector<MaterialGroup*>& 
OctreeNode::getMaterialGroups ()
{
	return m_MaterialGroups;
}

OctreeNode* 
OctreeNode::getChild(int i)
{
	return (this->m_Childs[i]);
}

vec3& 
OctreeNode::getBBMin (void)
{
	return m_bbMin;
}

vec3& 
OctreeNode::getBBMax (void)
{
	return m_bbMax;
}

void
OctreeNode::buildLists()
{
	
	m_VerticesList = new std::vector<vec3>;
	m_NormalsList = new std::vector<vec3>;
	m_TexCoordsList = new std::vector<vec3>;
	std::map<unsigned int, unsigned int> newIndicesMap;

	for (unsigned int i = 0; i < m_MaterialGroups.size(); i++){
		MaterialGroup* aMaterialGroup = m_MaterialGroups.at (i);

		
		std::vector<unsigned int>& indexesList	= aMaterialGroup->getIndexList();

		std::vector<vec3>& verticesPool = m_Master->getVerticesList();
		std::vector<vec3>& normalsPool = m_Master->getNormalsList();
		std::vector<vec3>& texCoordsPool = m_Master->getTexCoordsList();


		for (unsigned int i = 0; i < indexesList.size(); i++){
			if (0 == newIndicesMap.count (indexesList.at (i))){
				m_VerticesList->push_back (verticesPool. at(indexesList. at(i)));
				m_NormalsList->push_back (normalsPool. at(indexesList. at(i)));
				m_TexCoordsList->push_back (texCoordsPool. at(indexesList. at(i)));
				newIndicesMap[indexesList.at (i)] = m_VerticesList->size() - 1;
				indexesList.at (i) = m_VerticesList->size() - 1;
			} else {
				indexesList.at (i) = newIndicesMap[indexesList.at (i)];
			}
		}
	}

	for (int i = 0; i < 8; i++){
		if (0 != m_Childs[i]){
			m_Childs[i]->buildLists();
		}
	}
}

std::vector<vec3>& 
OctreeNode::getVerticesList()
{
	return (*m_VerticesList);
}

std::vector<vec3>& 
OctreeNode::getNormalsList()
{
	return (*m_NormalsList);
}

std::vector<vec3>& 
OctreeNode::getTexCoordsList()
{
	return (*m_TexCoordsList);
}


void
OctreeNode::appendMaterialGroup (IMaterialGroup& aMaterialGroup)
{
	std::string aMaterialGroupName = aMaterialGroup.getMaterialName ();

	std::vector<MaterialGroup*>::iterator materialGroupsIter = m_MaterialGroups.begin();;

	MaterialGroup *tempMaterialGroup = 0;

	bool addNew = false;

	//Need first to search for this material in this node 


	//if the vector of material groups is empty 
	if (true  == m_MaterialGroups.empty()){
		addNew = true;
		LOG_INFO("(%d, %d)The material group is empty", m_NodeId, m_NodeDepth);
	} else { //if not iterate the vector and search if the is a group with the same name
		tempMaterialGroup = (*materialGroupsIter);

		while (materialGroupsIter != m_MaterialGroups.end () && 
			tempMaterialGroup->getMaterialName () != aMaterialGroupName){
			tempMaterialGroup = (*materialGroupsIter);
			materialGroupsIter++;
		}
		if (materialGroupsIter == m_MaterialGroups.end ()){
			addNew = true;
		} else {
			addNew = false;
		}
	}

	//The material doesn't exist in this node and we need to add it
	if(true == addNew){

		MaterialGroup *aNewGroup;

		aNewGroup = new MaterialGroup;

		aNewGroup->setParent (m_Master);
		aNewGroup->setMaterialId (aMaterialGroup.getMaterialId ());
		aNewGroup->setMaterialName (aMaterialGroup.getMaterialName ());

		m_MaterialGroups.push_back (aNewGroup);
		tempMaterialGroup = aNewGroup;	
	} else {
		//PHONY
	}
	
	
	//Add the indices from this material to the already existent material

	std::vector<unsigned int>& aMaterialGroupIndexList = aMaterialGroup.getIndexList ();
	std::vector<unsigned int>& indexList = tempMaterialGroup->getIndexList ();

	std::vector<unsigned int>::iterator aMaterialGroupIndexListIter;
	
	for (aMaterialGroupIndexListIter = aMaterialGroupIndexList.begin ();
		aMaterialGroupIndexListIter != aMaterialGroupIndexList.end ();
		aMaterialGroupIndexListIter++){
			//LOG_INFO("The index: %d", *aMaterialGroupIndexListIter);
			indexList.push_back (*aMaterialGroupIndexListIter);
	}

	this->m_TriangleCount += indexList.size () / 3;
}

*/
