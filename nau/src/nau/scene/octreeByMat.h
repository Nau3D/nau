#ifndef OCTREEBYMAT_H
#define OCTREEBYMAT_H

#include "nau/scene/octreeByMatNode.h"
#include "nau/scene/iScene.h"
#include "nau/scene/camera.h"
#include "nau/math/vec3.h"
#include "nau/geometry/boundingBox.h"
#include "nau/clogger.h" /***MARK***/

#include <vector>
#include <string>

namespace nau {
	namespace loader {
		class CBOLoader;
	};
};

namespace nau
{

	namespace scene
	{


		class OctreeByMat {

			friend class nau::loader::CBOLoader;
			friend class OctreeByMatNode;

		public:
			OctreeByMat();

			virtual ~OctreeByMat();
			void setName(std::string name);
			std::string getName();
			void build (std::vector<std::shared_ptr<SceneObject>> &sceneObjects);
		
			void updateOctreeTransform(nau::math::mat4 &m_Transform);
			int getNumberOfVertices () { return 0; };

			//nau::math::vec3& getVertice (unsigned int v);
			void unitize(vec3 &center, vec3 &min, vec3 &max);
			void _compile (void);
			void _findVisibleSceneObjects (std::vector<std::shared_ptr<SceneObject>> *returnVector,
																nau::geometry::Frustum &aFrustum, 
																nau::scene::Camera &aCamera,
																bool conservative = false);

			void _getAllObjects (std::vector<std::shared_ptr<SceneObject>> *returnVector);

			void _place (std::shared_ptr<SceneObject> &aSceneObject);

			void getMaterialNames(std::set<std::string> *nameList);

		protected:
			std::shared_ptr<OctreeByMatNode> m_pOctreeRootNode;
			std::string m_Name;
			
		private:
			void _transformSceneObjects (std::vector<std::shared_ptr<SceneObject>> &sceneObjects);
			nau::geometry::BoundingBox _calculateBoundingBox (std::vector<std::shared_ptr<SceneObject>> &sceneObjects);


		};
	};
};

#endif // COCTREE_H
