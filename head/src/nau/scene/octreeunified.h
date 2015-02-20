#ifndef OCTREEUNIFIED_H
#define OCTREEUNIFIED_H

#include "nau/scene/iscenepartitioned.h"
#include "nau/geometry/frustum.h"
#include "nau/geometry/boundingbox.h"

namespace nau {

	namespace scene {

		class OctreeUnified : public IScenePartitioned
		{
		private:
			std::vector<SceneObject*> m_vReturnVector;
			SceneObject *m_SceneObject;

			nau::geometry::BoundingBox m_BoundingBox;

		protected:
			void updateSceneObjectTransforms();

		public:
			OctreeUnified(void);
			~OctreeUnified(void);

			void clear();

			virtual void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);

			virtual void build(void);
			virtual void compile(void);

			virtual nau::geometry::IBoundingVolume& getBoundingVolume(void);

			virtual void unitize();

			virtual void add(nau::scene::SceneObject *aSceneObject);

			virtual std::vector <SceneObject*>& findVisibleSceneObjects
				(nau::geometry::Frustum &aFrustum,
				Camera &aCamera,
				bool conservative = false);

			virtual std::vector<SceneObject*>& getAllObjects();
			virtual nau::scene::SceneObject* getSceneObject(std::string name);
			virtual nau::scene::SceneObject* getSceneObject(int index);

			virtual void getMaterialNames(std::set<std::string> *nameList);

			virtual nau::math::ITransform *getTransform();
			virtual void setTransform(nau::math::ITransform *t);
			virtual void transform(nau::math::ITransform *t);

			virtual std::string getType(void);

			/*
			* Statistical information
			*/

			int getNumTriangles() { return 0; };
			int getNumVertices() { return 0; };

			// By material
			int getNumTrianglesMat(int i) { return 0; };
			int getNumVerticesMat(int i) { return 0;; }

			// By Object
			int getNumTrianglesObject(int i) { return 0; };
			int getNumVerticesObject(int i) { return 0; };

		};
	};
};

#endif
