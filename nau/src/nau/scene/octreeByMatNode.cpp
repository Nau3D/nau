#include "nau/scene/octreeByMatNode.h"

#include "nau/clogger.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"
#include "nau/math/matrix.h"
#include "nau.h"



#include <assert.h>
#include <sstream>

using namespace nau::scene;
using namespace nau::render;
using namespace nau::material;
using namespace nau::geometry;
using namespace nau::math;

OctreeByMatNode::OctreeByMatNode () :
	//SceneObject(),
	m_pParent (0),
	m_ChildCount (0),
	m_Divided (false),
	m_NodeId (0),
	m_NodeDepth (0)
{
	for (int i = 0; i < 8; i++) {
		m_pChilds[i] = 0;
	}
//	m_Transform = new SimpleTransform;
}


OctreeByMatNode::OctreeByMatNode (OctreeByMatNode *parent, vec3 bbMin, vec3 bbMax, int nodeId, int nodeDepth) :
	m_ChildCount (0),
	m_Divided (false),
	m_NodeDepth (nodeDepth)
{
	m_pParent = std::shared_ptr<OctreeByMatNode>(parent);
	for (int i = 0; i < 8; i++) {
		m_pChilds[i] = 0;
	}

	m_NodeId = nodeId;

	m_BoundingVolume.set(bbMin, bbMax);
	m_TightBoundingVolume.set(bbMin, bbMax);

}

OctreeByMatNode::~OctreeByMatNode(void)
{
	//for (int i = 0; i < 8; i++) {
	//	delete m_pChilds[i];
	//}
}


void 
OctreeByMatNode::getMaterialNames(std::set<std::string> *nameList)
{
	for (auto &so: m_pLocalMeshes) 
		nameList->insert(so.first);

	for (int i = 0; i < 8 ; i++) {
		if (m_pChilds[i])
			m_pChilds[i]->getMaterialNames(nameList);
	}
}


void 
OctreeByMatNode::tightBoundingVolume() {

	for (auto &so: m_pLocalMeshes) 
		m_TightBoundingVolume.calculate ((so.second)->getRenderable()->getVertexData()->getDataOf (VertexData::GetAttribIndex(std::string("position"))));

	for (int i = TOPFRONTLEFT; i < ROOT; i++) {
	
		if (0 != m_pChilds[i]) {
			m_TightBoundingVolume.compound((const BoundingBox *)(&(m_pChilds[i]->m_TightBoundingVolume)));
		}
	}
}


void 
OctreeByMatNode::updateNodeTransform(nau::math::mat4 &t)
{
	for (auto &so : m_pLocalMeshes)
		so.second->updateGlobalTransform(t);

	for (int i = TOPFRONTLEFT; i < ROOT; i++) {
	
		if (0 != m_pChilds[i]) {
			m_pChilds[i]->updateNodeTransform(t);
		}
	}

}


void 
OctreeByMatNode::unitize(vec3 &center, vec3 &min, vec3 &max) {
	
	for (auto &so : m_pLocalMeshes)
		so.second->unitize(center, min, max);

	for (int i = OctreeByMatNode::TOPFRONTLEFT; i < OctreeByMatNode::ROOT;  i++) {
	
		if (0 != m_pChilds[i]) {
			m_pChilds[i]->unitize(center, min, max);
		}
	}
	tightBoundingVolume();
}


void
OctreeByMatNode::setName(std::string name) {

	m_Name = name;
}


std::string
OctreeByMatNode::getName() {

	return m_Name;
}



void 
OctreeByMatNode::setRenderable (std::shared_ptr<nau::render::IRenderable> &aRenderable)
{
	nau::resource::ResourceManager *rm = RESOURCEMANAGER;
	m_pLocalMeshes.clear();

	// first divide the renderable so that each renderable has only one material

	std::vector<std::shared_ptr<MaterialGroup>> &vMaterialGroups = aRenderable->getMaterialGroups();

	for (auto &pMaterialGroup: vMaterialGroups) {

		std::shared_ptr<nau::geometry::IndexData> &indexDataMaterialGroup = pMaterialGroup->getIndexData();
		std::shared_ptr<std::vector<unsigned int>> &vIndexData = indexDataMaterialGroup->getIndexData();

		if (vIndexData->size() > 0) {
		
			std::shared_ptr<SceneObject> so = SceneObjectFactory::Create("SimpleObject");
			so->setName(m_Name+"::"+pMaterialGroup->getMaterialName());

			m_pLocalMeshes[pMaterialGroup->getMaterialName()] = so;
			std::shared_ptr<IRenderable> &m = rm->createRenderable("Mesh", m_Name+"::"+pMaterialGroup->getMaterialName());

			m->addMaterialGroup(pMaterialGroup, aRenderable); 
			so->setRenderable(m);
			m_BoundingVolume.compound(so->getBoundingVolume());
			
		}
	}
	m_TightBoundingVolume = m_BoundingVolume;

	this->_split();

	// now gather all renderables from parent (not leaf nodes) local meshes to the root local mesh
	// the goal is to avoid having small buffers for rendering

	this->_unifyLocalMeshes();

}


void
OctreeByMatNode::_unifyLocalMeshes() {

	// recurse into children that are not leafs
	for (int index = TOPFRONTLEFT; index <= BOTTOMBACKRIGHT; index++) {
		if (m_pChilds[index] && m_pChilds[index]->m_ChildCount != 0)
				m_pChilds[index]->_unifyLocalMeshes();
	}

	// for each leaf
	for (int index = TOPFRONTLEFT; index <= BOTTOMBACKRIGHT; index++) {

		std::shared_ptr<OctreeByMatNode> &aux = m_pChilds[index];
		if (aux) {
			std::map<std::string, std::shared_ptr<SceneObject>>::iterator iter,iter2;
			iter = aux->m_pLocalMeshes.begin();

			while(  iter != aux->m_pLocalMeshes.end() ) {
		
				if (iter->second->getRenderable()->getNumberOfVertices()/3 < 8000) {

					if (!m_pLocalMeshes.count(iter->first))
						m_pLocalMeshes[iter->first] = SceneObjectFactory::Create("SimpleObject");

					m_pLocalMeshes[iter->first]->getRenderable()->addMaterialGroup((iter->second->getRenderable()->getMaterialGroups())[0], iter->second->getRenderable());

					aux->m_pLocalMeshes.erase(iter++);
				}
				else 
					++iter;
		
			}
		}
	}
}


void 
OctreeByMatNode::_split() {


	int vertexArrayPos = VertexData::GetAttribIndex(std::string("position"));
	int offSet;
	std::map<std::string, std::shared_ptr<SceneObject>>::iterator matIter;
	
	std::string name;

	// do the splitting for each material 
	int countSplits = (unsigned int)m_pLocalMeshes.size();
	matIter = m_pLocalMeshes.begin();
	for ( ; matIter != m_pLocalMeshes.end(); matIter++) {

		std::shared_ptr<MaterialGroup> tempMaterialGroup[9] = { 0 };
		std::shared_ptr<SceneObject> tempSO[9] = { 0 };

		name = matIter->first;
		std::shared_ptr<SceneObject> &s = matIter->second;

		std::shared_ptr<IRenderable> &r = m_pLocalMeshes[name]->getRenderable();
		std::shared_ptr<MaterialGroup> &pMaterialGroup = r->getMaterialGroups()[0];

		// if needs to be splitted
		if (r->getNumberOfVertices()/3 > MAXPRIMITIVES) {
			std::shared_ptr<VertexData> &vVertexData = r->getVertexData();
			std::shared_ptr<nau::geometry::IndexData> &VertexDataMaterialGroup = pMaterialGroup->getIndexData();
			std::shared_ptr<std::vector<unsigned int>> &vIndexData = VertexDataMaterialGroup->getIndexData();
			std::vector<unsigned int>::iterator indexIter;
			indexIter = vIndexData->begin();
			std::shared_ptr<std::vector<VertexData::Attr>> &vVertices = vVertexData->getDataOf(vertexArrayPos);
			// Octree are only implemented for triangles
			offSet = 3;// ((Mesh *)r)->getPrimitiveOffset();
			// for every primitive, split into temporary arrays
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
					tempMaterialGroup[index]->setMaterialName (pMaterialGroup->getMaterialName());
				}

				std::shared_ptr<std::vector<unsigned int>> &vTempIndexData =
					tempMaterialGroup[index]->getIndexData()->getIndexData();

				if (vTempIndexData) {
					std::shared_ptr<std::vector<unsigned int>> newIndexData = 
						std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>);
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

			for (int index = TOPFRONTLEFT; index <= ROOT; index++) {
				if (0 != tempMaterialGroup[index]) {
					if (0 == tempSO[index]) {
						stringstream s;
						s << r->getName() << "." << index;
						std::shared_ptr<IRenderable> &m = RESOURCEMANAGER->createRenderable("Mesh", s.str()); //new Mesh;
						tempSO[index] = SceneObjectFactory::Create("SimpleObject");
						tempSO[index]->setName(s.str());
						tempSO[index]->setRenderable(m);
					}
					Mesh *m = (Mesh* )&(tempSO[index]->getRenderable());
					m->addMaterialGroup (tempMaterialGroup[index],r);
				}
			}

 			for (int index = TOPFRONTLEFT; index <= BOTTOMBACKRIGHT; index++) {
				if (0 != tempSO[index]) {
					if (0 == m_pChilds[index]) {
						m_pChilds[index] = _createChild (index);
					}
					m_pChilds[index]->m_pLocalMeshes[name] = tempSO[index];
				}
			}

			if (0 != tempSO[ROOT]) {
				std::shared_ptr<SceneObject> &so = m_pLocalMeshes[name];
				RESOURCEMANAGER->removeRenderable(so->getName());
				m_pLocalMeshes[name] = tempSO[ROOT];
			}
		}
		else
			countSplits--;
	}
	if (countSplits)
		for (int index = TOPFRONTLEFT; index <= BOTTOMBACKRIGHT; index++) {
			if (m_pChilds[index])
				m_pChilds[index]->_split();
		}
}




std::string 
OctreeByMatNode::getType (void)
{
	return "OctreeByMatNode";
}


int
OctreeByMatNode::_octantFor (VertexAttrib& v)
{
	int octant = 8;

	const vec3 &BBCenter = m_BoundingVolume.getCenter();

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


std::shared_ptr<OctreeByMatNode> 
OctreeByMatNode::_createChild (int octant)
{
	vec3 bbMin;
	vec3 bbMax;

	const vec3 &LocalBBMin = m_BoundingVolume.getMin();
	const vec3 &LocalBBMax = m_BoundingVolume.getMax();
	const vec3 &LocalBBCenter = m_BoundingVolume.getCenter();

	switch (octant) {
		case OctreeByMatNode::TOPFRONTLEFT: {
			bbMin.x = LocalBBMin.x;
			bbMin.y = LocalBBCenter.y;
			bbMin.z = LocalBBCenter.z;
			
			bbMax.x = LocalBBCenter.x;
			bbMax.y = LocalBBMax.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeByMatNode::TOPFRONTRIGHT: {
			bbMin.x = LocalBBCenter.x;
			bbMin.y = LocalBBCenter.y;
			bbMin.z = LocalBBCenter.z;
			
			bbMax.x = LocalBBMax.x;
			bbMax.y = LocalBBMax.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeByMatNode::TOPBACKLEFT: {
			bbMin.x = LocalBBMin.x;
			bbMin.y = LocalBBCenter.y;
			bbMin.z = LocalBBMin.z;
			
			bbMax.x = LocalBBCenter.x;
			bbMax.y = LocalBBMax.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeByMatNode::TOPBACKRIGHT: {
			bbMin.x = LocalBBCenter.x;
			bbMin.y = LocalBBCenter.y;
			bbMin.z = LocalBBMin.z;
			
			bbMax.x = LocalBBMax.x;
			bbMax.y = LocalBBMax.y;
			bbMax.z = LocalBBCenter.z;
			break;
		}
		case OctreeByMatNode::BOTTOMFRONTLEFT: {
			bbMin.x = LocalBBMin.x; 
			bbMin.y = LocalBBMin.y;
			bbMin.z = LocalBBCenter.z;
			
			bbMax.x = LocalBBCenter.x;
			bbMax.y = LocalBBCenter.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeByMatNode::BOTTOMFRONTRIGHT: {
			bbMin.x = LocalBBCenter.x;
			bbMin.y = LocalBBMin.y;
			bbMin.z = LocalBBCenter.z;
			
			bbMax.x = LocalBBMax.x;
			bbMax.y = LocalBBCenter.y;
			bbMax.z = LocalBBMax.z;
			break;
		}
		case OctreeByMatNode::BOTTOMBACKLEFT: {
			bbMin.x = LocalBBMin.x; 
			bbMin.y = LocalBBMin.y;
			bbMin.z = LocalBBMin.z;
			
			bbMax.x = LocalBBCenter.x;
			bbMax.y = LocalBBCenter.y;
			bbMax.z = LocalBBCenter.z;
			break;
		}
		case OctreeByMatNode::BOTTOMBACKRIGHT: {
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
	return std::shared_ptr<OctreeByMatNode>(new OctreeByMatNode (this, bbMin, bbMax, octant, m_NodeDepth + 1));
}


void
OctreeByMatNode::_compile (void)
{

	for (auto &so: m_pLocalMeshes)  {

		so.second->getRenderable()->getVertexData()->compile();

		std::vector<std::shared_ptr<MaterialGroup>> &matGroups = so.second->getRenderable()->getMaterialGroups();

		for (auto &matGroupsIter: matGroups){
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
OctreeByMatNode::getAllSceneObjects (std::vector<std::shared_ptr<SceneObject>> *returnVector) {

	// for each scene object in the node, check if its in and add it
	for (auto &so: m_pLocalMeshes) {
		returnVector->push_back(so.second);
	}

	// recurse for children
	for (int i = TOPFRONTLEFT; i <= BOTTOMBACKRIGHT; i++) {
		if (0 != m_pChilds[i]) {
			m_pChilds[i]->getAllSceneObjects (returnVector);
		}
	}
}


void 
OctreeByMatNode::_findVisibleSceneObjects (std::vector<std::shared_ptr<SceneObject>> *returnVector,
																Frustum &aFrustum, 
																Camera &aCamera,
																bool conservative)
{
	int side;
	
	// check if the whole node is out
	side = aFrustum.isVolumeInside (&m_TightBoundingVolume, conservative);
	// if out then leave
	if (Frustum::OUTSIDE == side) {
		return;
	}

	// for each scene object in the node, check if its in and add it
	for (auto &so : m_pLocalMeshes) {
		side = aFrustum.isVolumeInside (so.second->getBoundingVolume());
		if (Frustum::OUTSIDE != side)
			returnVector->push_back(so.second);
	}

	// recurse for children
	for (int i = TOPFRONTLEFT; i <= BOTTOMBACKRIGHT; i++) {
		if (0 != m_pChilds[i]) {
			m_pChilds[i]->_findVisibleSceneObjects (returnVector, aFrustum, aCamera, conservative);	
		}
	}
}


std::shared_ptr<OctreeByMatNode> &
OctreeByMatNode::_getChild (int i) {
	return m_pChilds[i];
}


void 
OctreeByMatNode::_setChild (int i, std::shared_ptr<OctreeByMatNode> &aNode) {

	m_pChilds[i] = aNode;
}


void
OctreeByMatNode::_setParent (std::shared_ptr<OctreeByMatNode> &parent) {

	m_pParent = parent;
}


int
OctreeByMatNode::_getChildCount (void) {

	return m_ChildCount;
}


std::string
OctreeByMatNode::_genOctName (void) {

	char name[256];

	sprintf (name, "oct%d", rand());
	return name;
}

