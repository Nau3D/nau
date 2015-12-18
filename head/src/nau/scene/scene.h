#ifndef SCENE_H
#define SCENE_H

#include "nau/scene/iScene.h"
#include "nau/scene/camera.h"
#include "nau/scene/light.h"
#include "nau/geometry/frustum.h"
#include "nau/geometry/boundingBox.h"

namespace nau {

	namespace scene {

		class Scene : public IScene
		{
			friend class SceneFactory;

		protected:
			Scene(void);

			std::vector<SceneObject*> m_vReturnVector;
			std::vector<SceneObject*> m_SceneObjects;

			//bool m_Visible;

			nau::geometry::BoundingBox m_BoundingBox;
			void updateSceneObjectTransforms();

		public:
			~Scene (void);

			void clear();

			virtual void eventReceived(const std::string &sender, const std::string &eventType, 
				const std::shared_ptr<IEventData> &evt);
			virtual void add (nau::scene::SceneObject *aSceneObject);

			virtual void build (void);
			virtual void compile (void);

			virtual nau::geometry::IBoundingVolume& getBoundingVolume (void);

			virtual void unitize();

			virtual std::vector <SceneObject*>& findVisibleSceneObjects 
																(nau::geometry::Frustum &aFrustum, 
																Camera &aCamera,
																bool conservative = false);
			virtual std::vector<SceneObject*>& getAllObjects ();
			virtual nau::scene::SceneObject* getSceneObject (std::string name);
			virtual nau::scene::SceneObject* getSceneObject (int index);

			virtual const std::set<std::string> &getMaterialNames();

			virtual nau::math::mat4 &getTransform();
			virtual void setTransform(nau::math::mat4 &t);
			virtual void transform(nau::math::mat4 &t);

		};
	};
};

#endif
