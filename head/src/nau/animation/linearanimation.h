#ifndef LINEARANIMATION_H
#define LINEARANIMATION_H

#include "nau/animation/ianimation.h"

#include "nau/scene/sceneobject.h"
#include "nau/math/vec3.h"

namespace nau
{
	namespace animation
	{
		class LinearAnimation : public IAnimation
		{
		public:
			LinearAnimation (nau::scene::ISceneObject *aObject, nau::math::vec3 start, nau::math::vec3 end);
			LinearAnimation (nau::scene::ISceneObject *aObject, nau::math::vec3 end);

			void step (float deltaT);
			bool isFinished (void);

			~LinearAnimation(void);
		private:
			nau::scene::ISceneObject *m_SceneObject;
			nau::math::vec3 m_StartPos;
			nau::math::vec3 m_LineVector;
			nau::math::vec3 m_CurrentPos;

			float m_LocalTime;
		};
	};
};

#endif //LINEARANIMATION_H
