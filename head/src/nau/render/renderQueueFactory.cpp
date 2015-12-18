#include "nau/render/renderQueueFactory.h"
#include "nau/render/materialSortRenderQueue.h"

using namespace nau::render;

IRenderQueue*
RenderQueueFactory::create (std::string queueType)
{
	if ("MaterialSort" == queueType) {
		return new MaterialSortRenderQueue;
	}
	return 0;
}
