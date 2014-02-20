#include <nau/render/quadpass.h>

#include <nau.h>

using namespace nau::render;
using namespace nau::geometry;

QuadPass::QuadPass (const std::string &name) :
	Pass (name),
	m_QuadObject (0)
{
	m_ClassName = "quad";
	m_QuadObject = new Quad;
	//m_MaterialMap["__Quad"] = MaterialID(DEFAULTMATERIALLIBNAME, "__Quad");
	//std::vector<nau::material::IMaterialGroup*>& matGroups = m_QuadObject->_getRenderablePtr()->getMaterialGroups();

	//std::vector<nau::material::IMaterialGroup*>::iterator matGroupsIter;

	//matGroupsIter = matGroups.begin();

	//for ( ; matGroupsIter != matGroups.end(); matGroupsIter++) {
	//	(*matGroupsIter)->setMaterialName ("__Quad");
	//}
	//m_MaterialMap["__Quad"] = MaterialID(name,"quad");
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

	RENDERER->setMatrixMode (IRenderer::PROJECTION_MATRIX);
	RENDERER->pushMatrix();
	RENDERER->loadIdentity();

	RENDERER->setMatrixMode (IRenderer::VIEW_MATRIX);
	RENDERER->pushMatrix();
	RENDERER->loadIdentity();	

	RENDERER->setMatrixMode (IRenderer::MODEL_MATRIX);
	RENDERER->pushMatrix();
	RENDERER->loadIdentity();


	if (m_pViewport != NULL)
		RENDERER->setViewport(m_pViewport);

	prepareBuffers();

#if (NAU_CORE_OPENGL == 0)
	RENDERER->deactivateLighting();
	RENDERER->enableTexturing();
#endif
}

void
QuadPass::restore (void)
{
	if (0 != m_RenderTarget && true == m_UseRT) {
		m_RenderTarget->unbind();
	}

	RENDERER->setMatrixMode (IRenderer::PROJECTION_MATRIX);
	RENDERER->popMatrix();

	RENDERER->setMatrixMode (IRenderer::VIEW_MATRIX);
	RENDERER->popMatrix();

	RENDERER->setMatrixMode (IRenderer::MODEL_MATRIX);
	RENDERER->popMatrix();

#if (NAU_CORE_OPENGL == 0)
	RENDERER->disableTexturing();
#endif
}

void 
QuadPass::doPass (void)
{

	RENDERMANAGER->clearQueue();
	RENDERMANAGER->addToQueue (m_QuadObject, m_MaterialMap);
	RENDERMANAGER->processQueue();

}

