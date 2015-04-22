#include "nau/render/opengl/glmaterialgroup.h"

#include "nau/render/opengl/glindexarray.h"
#include "nau/render/opengl/glvertexarray.h"


#include <GL/glew.h>

using namespace nau::render::opengl;

GLMaterialGroup::GLMaterialGroup(nau::render::IRenderable *parent, std::string materialName) : 
	MaterialGroup(parent, materialName), 
	m_VAO(0) {

}

GLMaterialGroup::~GLMaterialGroup() {

	if (m_VAO)
		glDeleteVertexArrays(1, &m_VAO);
}


void
GLMaterialGroup::compile() {

	if (m_VAO)
		return;

	VertexData &v = m_Parent->getVertexData();

	if (!v.isCompiled())
		v.compile();

	if (!m_IndexData->isCompiled())
		m_IndexData->compile();

	glGenVertexArrays(1, &m_VAO);
	glBindVertexArray(m_VAO);

	v.bind();
	m_IndexData->bind();

	glBindVertexArray(0);
	v.unbind();
	m_IndexData->unbind();
}


void
GLMaterialGroup::resetCompilationFlag() {

	if (!m_VAO)
		return;

	glDeleteVertexArrays(1, &m_VAO);
	m_VAO = 0;

	VertexData &v = m_Parent->getVertexData();
	v.resetCompilationFlag();

	m_IndexData->resetCompilationFlag();
}


bool
GLMaterialGroup::isCompiled() {

	return (m_VAO != 0);
}


void
GLMaterialGroup::bind() {

	glBindVertexArray(m_VAO);
}


void 
GLMaterialGroup::unbind() {

	glBindVertexArray(0);
}


unsigned int
GLMaterialGroup::getVAO() {

	return m_VAO;
}