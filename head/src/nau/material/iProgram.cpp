#include "nau/material/iProgram.h"


#ifdef NAU_OPENGL
#include "nau/render/opengl/glProgram.h"
#endif

using namespace nau::render;

#if NAU_OPENGL_VERSION >= 430
std::string IProgram::ShaderNames[IProgram::SHADER_COUNT] = 
	{"Vertex", "Geometry", "Tess Control", "Tess Eval", "Fragment", "Compute"};
#elif NAU_OPENGL_VERSION >= 400
std::string IProgram::ShaderNames[IProgram::SHADER_COUNT] = 
	{"Vertex", "Geometry", "Tess Control", "Tess Eval", "Fragment"};
#elif NAU_OPENGL_VERSION >= 320
std::string IProgram::ShaderNames[IProgram::SHADER_COUNT] = 
	{"Vertex", "Geometry", "Fragment"};
#else
std::string IProgram::ShaderNames[IProgram::SHADER_COUNT] = 
	{"Vertex", "Fragment"};
#endif

using namespace nau::render;

IProgram*
IProgram::create (void) 
{
#ifdef NAU_OPENGL
	return new GLProgram;
#elif NAU_DIRECTX
	return new DXProgram;
#endif
}
