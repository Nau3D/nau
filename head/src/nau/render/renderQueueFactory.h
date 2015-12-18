#ifndef RENDERQUEUEFACTORY_H
#define RENDERQUEUEFACTORY_H

#include <string>

#include "nau/render/iRenderQueue.h"

namespace nau
{
	namespace render
	{
		class RenderQueueFactory
		{
		public:
			static IRenderQueue* create (std::string renderType);
		private:
			RenderQueueFactory(void);
			~RenderQueueFactory(void);
		};
	};
};

#endif //RENDERQUEUEFACTORY_H

