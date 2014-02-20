#ifndef SCENEPOSEOBJECT_H
#define SCENEPOSEOBJECT_H

#include <nau/scene/sceneobject.h>

namespace nau
{
	namespace scene
	{
		class ScenePoseObject : public SceneObject
		{
		public:
			ScenePoseObject (void);
			virtual ~ScenePoseObject(void);

			virtual void unitize(float min, float max);
			virtual bool isStatic();
			virtual void setStaticCondition(bool aCondition);

			virtual const nau::geometry::IBoundingVolume* getBoundingVolume();
			virtual void setBoundingVolume (nau::geometry::IBoundingVolume *b);
			virtual const nau::math::ITransform& getTransform();
			//virtual void setTransform (nau::math::ITransform *t);

			virtual void burnTransform (void);
			
			virtual void writeSpecificData (std::fstream &f);
			virtual void readSpecificData (std::fstream &f);

			virtual std::string getType (void);

		protected:
			void calculateBoundingVolume (void);
		};
	};
};

#endif // SCENEOBJECT_H
