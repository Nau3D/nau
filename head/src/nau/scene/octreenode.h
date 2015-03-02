#ifndef OCTREENODE_H
#define OCTREENODE_H

#include <vector>

#include "nau/scene/sceneobject.h"
#include "nau/scene/camera.h"
#include "nau/material/materialgroup.h"
#include "nau/render/irenderable.h"
#include "nau/geometry/boundingbox.h"
#include "nau/geometry/mesh.h"
#include "nau/geometry/frustum.h"
#include "nau/math/vec3.h"

namespace nau
{
	namespace scene
	{

		class OctreeNode : public SceneObject
		{
//			friend class boost::serialization::access;
			friend class Octree;
			
		protected:
			
			static const int MAXPRIMITIVES = 25000;

			enum {
			  TOPFRONTLEFT = 0,
			  TOPFRONTRIGHT,
			  TOPBACKLEFT,
			  TOPBACKRIGHT,
			  BOTTOMFRONTLEFT,
			  BOTTOMFRONTRIGHT,
			  BOTTOMBACKLEFT,
			  BOTTOMBACKRIGHT,
			  ROOT
			};

			OctreeNode* m_pParent;
			OctreeNode* m_pChilds[8];

			int m_ChildCount;
			bool m_Divided;
			int m_NodeId;
			int m_NodeDepth;

			nau::geometry::Mesh *m_pLocalMesh;

		//	nau::geometry::BoundingBox m_BoundingBox;

		public:
			OctreeNode ();
			
			OctreeNode (OctreeNode *parent, nau::geometry::IBoundingVolume *boundingBox, int nodeId = 0, int nodeDepth = 0);
			void updateNodeTransform(nau::math::mat4 &t);
			//void addRenderable (nau::render::IRenderable *aRenderable);
			void setRenderable (nau::render::IRenderable *renderable);

			void getMaterialNames(std::set<std::string> *nameList);

			virtual std::string getType (void);

			virtual void writeSpecificData (std::fstream &f);
			virtual void readSpecificData (std::fstream &f);
			void tightBoundingVolume();
			void OctreeNode::unitize(float min, float max);
			
			virtual ~OctreeNode(void);


		protected:
			void _compile (void);
			void _findVisibleSceneObjects (std::vector<nau::scene::SceneObject*> &m_vReturnVector,
						nau::geometry::Frustum &aFrustum, 
						nau::scene::Camera &aCamera,
						bool conservative = false);
			void resetCounter();	
			static int counter;
			

			OctreeNode* _getChild (int i);
			void _setParent (OctreeNode *parent);
			void _setChild (int i, OctreeNode *aNode);
			int _getChildCount (void);

		private:
			int _octantFor (VertexData::Attr& v);
			OctreeNode* _createChild (int octant);
			std::string _genOctName (void);


		};
	};
};
#endif //COCTREENODE_H
