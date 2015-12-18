#include "nau/material/IUniformBlock.h"

#include "nau/config.h"
#ifdef NAU_OPENGL
#include "nau/render/opengl/glUniformBlock.h"
#endif


using namespace nau::material;

IUniformBlock *
IUniformBlock::Create(std::string &name, unsigned int size) {

#ifdef NAU_OPENGL
	return new GLUniformBlock(name, size);
#endif
}