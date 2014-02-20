#include <nau/loader/textureloader.h>

#include <nau/loader/deviltextureloader.h>

using namespace nau::loader;

TextureLoader*
TextureLoader::create (void)
{
	return new DevILTextureLoader();
}
