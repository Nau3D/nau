#include "nau/render/renderqueuefactory.h"
#include "nau/render/materialsortrenderqueue.h"

using namespace nau::render;

IRenderQueue*
RenderQueueFactory::create (std::string queueType)
{
	if ("MaterialSort" == queueType) {
		return new MaterialSortRenderQueue;
	}
	return 0;
}
