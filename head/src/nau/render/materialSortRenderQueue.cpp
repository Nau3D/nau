#include "nau/render/materialSortRenderQueue.h"

#include "nau.h"
#include "nau/render/iRenderable.h"
#include "nau/geometry/boundingBox.h"

#include "nau/debug/profile.h"

#pragma warning(disable : 4503)

using namespace nau::render;
using namespace nau::scene;
using namespace nau::material;
using namespace nau::math;
using namespace nau;

typedef std::pair<MaterialGroup*, mat4*> pair_MatGroup_Transform;

MaterialSortRenderQueue::MaterialSortRenderQueue(void)
{
}

MaterialSortRenderQueue::~MaterialSortRenderQueue(void)
{
}

void 
MaterialSortRenderQueue::clearQueue (void)
{
	std::map<int, std::map<Material*, std::vector<pair_MatGroup_Transform >* >* >::iterator mapIter;

	mapIter = m_RenderQueue.begin();

	for ( ; mapIter != m_RenderQueue.end(); mapIter++) {
		std::map <Material*, std::vector<pair_MatGroup_Transform >* > *aMap;
		std::map <Material*, std::vector<pair_MatGroup_Transform >* >::iterator mapIter2;

		aMap = (*mapIter).second;

			mapIter2 = aMap->begin();
			for ( ; mapIter2 != aMap->end(); mapIter2++) {
				//if ((*mapIter2).second != NULL)
					delete (*mapIter2).second;
			}

			delete aMap;
	}

	m_RenderQueue.clear(); /***MARK***/ //Possible memory leak
}

void 
MaterialSortRenderQueue::addToQueue (SceneObject* aObject,
									 std::map<std::string, MaterialID> &materialMap)
{
	PROFILE ("Queue add");

	int order;
	Material* aMaterial = 0;
	MaterialLibManager *m = NAU->getMaterialLibManager();

	IRenderable &aRenderable = aObject->getRenderable();

	std::vector<MaterialGroup*> vMaterialGroups = aRenderable.getMaterialGroups();
	std::vector<MaterialGroup*>::iterator groupIter;
	groupIter = vMaterialGroups.begin();

	for ( ; groupIter != vMaterialGroups.end(); groupIter++ ) {
		MaterialGroup *aGroup = (*groupIter);
		IndexData &indexData = aGroup->getIndexData();
		{
			PROFILE ("Get material");
			aMaterial = materialMap[aGroup->getMaterialName()].m_MatPtr;
		}
		order = aMaterial->getState()->getPropi(IState::ORDER);
		if ((order >= 0) && (0 != aMaterial) && (true == aMaterial->isEnabled())) {

			if (0 == m_RenderQueue.count (order)){
				m_RenderQueue[order] = new std::map <Material*, std::vector<pair_MatGroup_Transform >* >;
			}
			std::map<Material*, std::vector<pair_MatGroup_Transform >* > *materialMap = m_RenderQueue[order];

			if (0 == materialMap->count (aMaterial)) {
				(*materialMap)[aMaterial] = new std::vector<pair_MatGroup_Transform >;
			}
			std::vector<pair_MatGroup_Transform > *matGroupVec = (*materialMap)[aMaterial];
			matGroupVec->push_back (pair_MatGroup_Transform(aGroup, aObject->_getTransformPtr()));
		}
	}

// ADD BOUNDING BOXES TO QUEUE
#ifdef NAU_RENDER_FLAGS
	if (NAU->getRenderFlag(Nau::BOUNDING_BOX_RENDER_FLAG)) {

		Profile("Enqueue Bounding Boxes");

		groupIter = nau::geometry::BoundingBox::getGeometry()->getMaterialGroups().begin();

		for ( ; groupIter != nau::geometry::BoundingBox::getGeometry()->getMaterialGroups().end(); groupIter++ ) {
			MaterialGroup *aGroup = (*groupIter);;
			Material *aMaterial = MATERIALLIBMANAGER->getMaterial(DEFAULTMATERIALLIBNAME, aGroup->getMaterialName());
			mat4 *trans = &((nau::geometry::BoundingBox *)(aObject->getBoundingVolume()))->getTransform();
			if (0 == m_RenderQueue.count (0)){
					m_RenderQueue[0] = new std::map <Material*, std::vector<pair_MatGroup_Transform >* >;
				}
				std::map<Material*, std::vector<pair_MatGroup_Transform >* > *materialMap = m_RenderQueue[aMaterial->getState()->getPropi(IState::ORDER)];

				if (0 == materialMap->count (aMaterial)) {
					(*materialMap)[aMaterial] = new std::vector<pair_MatGroup_Transform >;
				}
				std::vector<pair_MatGroup_Transform > *matGroupVec = (*materialMap)[aMaterial];
				nau::geometry::BoundingBox *bb = (nau::geometry::BoundingBox *)(aObject->getBoundingVolume());
				matGroupVec->push_back( pair_MatGroup_Transform(aGroup, &(bb->getTransform())));
		}
	}
#endif
}


void
MaterialSortRenderQueue::processQueue (void)
{
	PROFILE ("Process queue");

	IRenderer *renderer = RENDERER;

	std::map <int, std::map<Material*, std::vector<pair_MatGroup_Transform >* >* >::iterator renderQueueIter;

	renderQueueIter = m_RenderQueue.begin();

	for (; renderQueueIter != m_RenderQueue.end(); ++renderQueueIter) {
		std::map<Material*, std::vector<pair_MatGroup_Transform >* >::iterator materialMapIter;

		materialMapIter = (*renderQueueIter).second->begin();

		for (; materialMapIter != (*renderQueueIter).second->end(); materialMapIter++) {
			Material *aMat = (*materialMapIter).first;
			
			{
				PROFILE ("Material prepare");
				aMat->prepare();
			}
			
			std::vector<pair_MatGroup_Transform >::iterator matGroupsIter;

			matGroupsIter = (*materialMapIter).second->begin();
			
			{
				PROFILE ("Geometry rendering");
				for (; matGroupsIter != (*materialMapIter).second->end(); ++matGroupsIter) {
					bool b = (*matGroupsIter).second->isIdentity();
					if (!b) {
						renderer->pushMatrix(IRenderer::MODEL_MATRIX);
						renderer->applyTransform(IRenderer::MODEL_MATRIX, *(*matGroupsIter).second);
						aMat->setUniformValues();
						aMat->setUniformBlockValues();
					}
					{	PROFILE("Draw");
						renderer->drawGroup ((*matGroupsIter).first);
					}
					if (!b)
						renderer->popMatrix(IRenderer::MODEL_MATRIX);
				}
			}
			aMat->restore();
		}
	}
}

