#ifndef IRENDERALGORITHM
#define IRENDERALGORITHM

#include "nau/scene/iscene.h"
#include "nau/render/irenderer.h"

namespace nau 
{
	namespace render
	{
		class IRenderAlgorithm
		{
		public:
			virtual void renderScene (nau::scene::IScene* aScene) = 0;
			virtual void setRenderer (nau::render::IRenderer *aRenderer) = 0;
			virtual void init (void) = 0;

			virtual void externCommand (char keyCode) = 0;
		public:
			virtual ~IRenderAlgorithm(void) {};
		};
	};
};

#endif
