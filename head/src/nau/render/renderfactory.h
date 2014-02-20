#ifndef RENDERFACTORY
#define RENDERFACTORY

#include <nau/render/irenderer.h>

namespace nau
{
	namespace render
	{
		class RenderFactory
		{
		public:
			static IRenderer* create (void);
		private:
			RenderFactory (void) {};
		};
	};
};

#endif //RENDERFACTORY
