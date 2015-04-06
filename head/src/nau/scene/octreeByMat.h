#ifndef OCTREEBYMAT_H
#define OCTREEBYMAT_H

#include "nau/scene/octreeByMatnode.h"
#include "nau/scene/iscene.h"
#include "nau/scene/camera.h"
#include "nau/math/vec3.h"
#include "nau/geometry/boundingbox.h"
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
			void build (std::vector<nau::scene::SceneObject*> &sceneObjects);
		
			void updateOctreeTransform(nau::math::mat4 &m_Transform);
			int getNumberOfVertices () { return 0; };

			nau::math::vec3& getVertice (unsigned int v);
			void unitize(vec3 &center, vec3 &min, vec3 &max);
			void _compile (void);
			void _findVisibleSceneObjects (std::vector<nau::scene::SceneObject*> &m_vReturnVector,
																nau::geometry::Frustum &aFrustum, 
																nau::scene::Camera &aCamera,
																bool conservative = false);

			void _getAllObjects (std::vector<nau::scene::SceneObject*> &m_vReturnVector);

			void _place (nau::scene::SceneObject *aSceneObject);

			void getMaterialNames(std::set<std::string> *nameList);

		protected:
			OctreeByMatNode *m_pOctreeRootNode;
			std::string m_Name;
			std::vector<nau::scene::SceneObject*> m_vReturnVector;
			
		private:
			void _transformSceneObjects (std::vector<SceneObject*> &sceneObjects);
			nau::geometry::BoundingBox _calculateBoundingBox (std::vector<SceneObject*> &sceneObjects);


		};
	};
};

#endif // COCTREE_H
