#include "nau/render/opengl/glMaterialArrayOfTextures.h"

#include "nau/render/iAPISupport.h"

#include <glbinding/gl/gl.h>
using namespace gl;


using namespace nau::render;
using namespace nau::material;

bool
GLMaterialArrayOfTextures::Init() {

	//Attribs.setDefault("TYPE", new int(GL_ATOMIC_COUNTER_BUFFER));

	Attribs.listAdd("TYPE", "ATOMIC_COUNTER", (int)GL_ATOMIC_COUNTER_BUFFER, IAPISupport::BUFFER_ATOMICS);
	Attribs.listAdd("TYPE", "SHADER_STORAGE", (int)GL_SHADER_STORAGE_BUFFER, IAPISupport::BUFFER_SHADER_STORAGE);
	Attribs.listAdd("TYPE", "UNIFORM", (int)GL_UNIFORM_BUFFER, IAPISupport::BUFFER_UNIFORM);

	return true;
}


bool GLMaterialArrayOfTextures::Inited = Init();


GLMaterialArrayOfTextures::GLMaterialArrayOfTextures(): IMaterialArrayOfTextures() {

}


GLMaterialArrayOfTextures::~GLMaterialArrayOfTextures() {

}


void
GLMaterialArrayOfTextures::bind() {

	int id = m_TextureArray->getPropi(IArrayOfTextures::BUFFER_ID);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_IntProps[BINDING_POINT], id);
}


void
GLMaterialArrayOfTextures::unbind() {

	glBindBufferBase(GL_UNIFORM_BUFFER, m_IntProps[BINDING_POINT], 0);
}
