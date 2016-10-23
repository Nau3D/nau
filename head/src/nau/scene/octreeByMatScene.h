#ifndef OCTREEBYMATSCENE_H
#define OCTREEBYMATSCENE_H

#include "nau/scene/iScenePartitioned.h"
#include "nau/scene/octreeByMat.h"
#include "nau/geometry/frustum.h"
#include "nau/geometry/boundingBox.h"


namespace nau {
	namespace loader {
		class CBOLoader;
	};
};

namespace nau {

	namespace scene {

		class OctreeByMatScene : public IScenePartitioned
		{
			friend class SceneFactory;

		public:
			friend class nau::loader::CBOLoader;
		private:
			std::vector<std::shared_ptr<SceneObject>> m_SceneObjects;

			nau::geometry::BoundingBox m_BoundingBox;

		protected:
			OctreeByMatScene (void);

			void updateSceneObjectTransforms();
			OctreeByMat *m_pGeometry;

		public:
			~OctreeByMatScene (void);

			//void clear();

			virtual void eventReceived(const std::string &sender, const std::string &eventType, 
				const std::shared_ptr<IEventData> &evt);

			virtual void build (void);
			virtual void compile (void);

			virtual nau::geometry::IBoundingVolume& getBoundingVolume (void);

			virtual void unitize();

			virtual void add(std::shared_ptr<SceneObject> &aSceneObject);
			virtual void findVisibleSceneObjects(std::vector<std::shared_ptr<SceneObject>> *v,
				nau::geometry::Frustum &aFrustum,
				Camera &aCamera,
				bool conservative = false);
			virtual void getAllObjects(std::vector<std::shared_ptr<SceneObject>> *);
			virtual std::shared_ptr<SceneObject> &getSceneObject(std::string name);
			virtual std::shared_ptr<SceneObject> &getSceneObject(int index);

			virtual const std::set<std::string> &getMaterialNames();

			virtual nau::math::mat4 &getTransform();
			virtual void setTransform(nau::math::mat4 &t);
			virtual void transform(nau::math::mat4 &t);


			/*
			 * Statistical information
			 */

			int getNumTriangles() { return 0; };
			int getNumVertices() { return 0; };
		
			// By material
			int getNumTrianglesMat(int i) { return 0; };
			int getNumVerticesMat(int i) { return 0; ;}

			// By Object
			int getNumTrianglesObject(int i) { return 0; };
			int getNumVerticesObject(int i) { return 0; };

			//virtual void scale(float factor) = 0;
			//virtual void translate(float x, float y, float z);
			//virtual void rotate(float ang, float ax, float ay, float az) = 0;
			//void show (void);			
			//void hide (void);
			//bool isVisible (void);
		};
	};
};

#endif
