#ifndef OCTREE_H
#define OCTREE_H

#include "nau/scene/octreeNode.h"
#include "nau/scene/iScene.h"
#include "nau/scene/camera.h"
#include "nau/math/vec3.h"
#include "nau/geometry/boundingBox.h"
#include "nau/clogger.h" /***MARK***/

#include <vector>
#include <string>

namespace nau
{

	namespace scene
	{

		//class COctreeNode;
		class Octree {

			friend class OctreeNode;
			//friend class boost::serialization::access;

		public:
			Octree();

			virtual ~Octree();

			void build (std::vector<nau::scene::SceneObject*> &sceneObjects);
		
			void updateOctreeTransform(nau::math::mat4  &m_Transform);
			int getNumberOfVertices () { return 0; };

			nau::math::vec3& getVertice (unsigned int v);
			void unitize(vec3 &vCenter, vec3 &vMin, vec3 &vMax);
			void _compile (void);
			void _findVisibleSceneObjects (std::vector<nau::scene::SceneObject*> &m_vReturnVector,
																nau::geometry::Frustum &aFrustum, 
																nau::scene::Camera &aCamera,
																bool conservative = false);

			void _getAllObjects (std::vector<nau::scene::SceneObject*> &m_vReturnVector);

			void _place (nau::scene::SceneObject *aSceneObject);

			void getMaterialNames(std::set<std::string> *nameList);

		private:
			//void renderOctreeNode (COctreeNode* aNode, IWorld& aWorld, bool testAgainstFrustum);

		private:
			OctreeNode *m_pOctreeRootNode;
			//std::vector<nau::material::IMaterialGroup*> m_MaterialGroups;
			std::vector<nau::scene::SceneObject*> m_vReturnVector;
			
		private:
			void _transformSceneObjects (std::vector<SceneObject*> &sceneObjects);
			nau::geometry::BoundingBox _calculateBoundingBox (std::vector<SceneObject*> &sceneObjects);

			//boot serialization interface
		private:
			template<class Archive>
			void serialize (Archive &ar, const unsigned int version)
			{
			}

		};
	};
};

#endif // COCTREE_H
