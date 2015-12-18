#include "nau/render/renderFactory.h"
#include "nau/render/opengl/glRenderer.h"
#include "nau/config.h"

using namespace nau::render;

IRenderer* 
RenderFactory::create (void)
{
#ifdef NAU_OPENGL
	return new GLRenderer;
#elif NAU_DIRECTX
	return new DXRenderer;
#else
	return 0;
#endif
}
