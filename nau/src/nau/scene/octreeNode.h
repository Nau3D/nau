#ifndef OCTREENODE_H
#define OCTREENODE_H

#include <vector>

#include "nau/scene/sceneObject.h"
#include "nau/scene/camera.h"
#include "nau/material/materialGroup.h"
#include "nau/render/iRenderable.h"
#include "nau/geometry/boundingBox.h"
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

			std::shared_ptr<OctreeNode> m_pParent;
			std::shared_ptr<OctreeNode> m_pChilds[8];

			int m_ChildCount;
			bool m_Divided;
			int m_NodeId;
			int m_NodeDepth;

			std::shared_ptr<nau::render::IRenderable> m_pLocalMesh;

		//	nau::geometry::BoundingBox m_BoundingBox;

		public:
			OctreeNode ();
			
			OctreeNode (std::shared_ptr<OctreeNode> parent, nau::geometry::IBoundingVolume *boundingBox, int nodeId = 0, int nodeDepth = 0);
			void updateNodeTransform(nau::math::mat4 &t);
			//void addRenderable (nau::render::IRenderable *aRenderable);
			void setRenderable (std::shared_ptr<nau::render::IRenderable> &renderable);

			void getMaterialNames(std::set<std::string> *nameList);

			virtual std::string getType (void);

			virtual void writeSpecificData (std::fstream &f);
			virtual void readSpecificData (std::fstream &f);
			void tightBoundingVolume();
			void unitize(vec3 &center, vec3 &min, vec3 &max);
			
			virtual ~OctreeNode(void);

			void eventReceived(const std::string & sender, const std::string & eventType, 
				const std::shared_ptr<IEventData>& evt);


		protected:
			void _compile (void);
			void _findVisibleSceneObjects (std::vector<std::shared_ptr<SceneObject>> *v,
						nau::geometry::Frustum &aFrustum, 
						nau::scene::Camera &aCamera,
						bool conservative = false);
			void resetCounter();	
			static int counter;
			

			std::shared_ptr<OctreeNode> &_getChild (int i);
			void _setParent (std::shared_ptr<OctreeNode> &parent);
			void _setChild (int i, std::shared_ptr<OctreeNode> &aNode);
			int _getChildCount (void);

		private:
			int _octantFor (VertexData::Attr& v);
			std::shared_ptr<OctreeNode> &_createChild (int octant);
			std::string _genOctName (void);

			std::shared_ptr<OctreeNode> m_Temp;


		};
	};
};
#endif //COCTREENODE_H
