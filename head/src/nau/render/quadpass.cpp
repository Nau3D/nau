#include "nau/render/quadpass.h"

#include "nau.h"

using namespace nau::render;
using namespace nau::geometry;

QuadPass::QuadPass (const std::string &name) :
	Pass (name),
	m_QuadObject (0)
{
	m_ClassName = "quad";
	m_QuadObject = new Quad;
}


QuadPass::~QuadPass(void)
{
	delete m_QuadObject;
}


void
QuadPass::prepare (void)
{
	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->bind();
	}

	RENDERER->pushMatrix(IRenderer::PROJECTION_MATRIX);
	RENDERER->loadIdentity(IRenderer::PROJECTION_MATRIX);

	RENDERER->pushMatrix(IRenderer::VIEW_MATRIX);
	RENDERER->loadIdentity(IRenderer::VIEW_MATRIX);

	RENDERER->pushMatrix(IRenderer::MODEL_MATRIX);
	RENDERER->loadIdentity(IRenderer::MODEL_MATRIX);


	if (m_Viewport != NULL)
		RENDERER->setViewport(m_Viewport);

	prepareBuffers();
}


void
QuadPass::restore (void)
{
	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->unbind();
	}
	RENDERER->popMatrix(IRenderer::PROJECTION_MATRIX);
	RENDERER->popMatrix(IRenderer::VIEW_MATRIX);
	RENDERER->popMatrix(IRenderer::MODEL_MATRIX);
}


void 
QuadPass::doPass (void)
{
	RENDERMANAGER->clearQueue();
	RENDERMANAGER->addToQueue (m_QuadObject, m_MaterialMap);
	RENDERMANAGER->processQueue();

}

