#include <nau/render/opengl/glrendertarget.h>
#include <nau.h>

#include <assert.h>

using namespace nau::render;


GLRenderTarget::GLRenderTarget(std::string name, unsigned int width, unsigned int height) :
	RenderTarget (name, width, height),
	
	m_DepthBuffer (0),
//	m_RTCount (0),
	m_NoDrawAndRead (false),
	m_RenderTargets(0)
{
	m_Samples = 0;
	//for (int i = 0; i < RenderTarget::MAXFBOs; ++i) {
	//	m_RenderTargets[i] = -1;
	//}

	glGenFramebuffers (1, &m_Id);
	glGenRenderbuffers (1, &m_DepthBuffer);
	glBindRenderbuffer (GL_RENDERBUFFER, m_DepthBuffer);
	glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_Width, m_Height);

	bind();
	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_DepthBuffer);
	unbind();

	glBindRenderbuffer (GL_RENDERBUFFER, 0);
}


GLRenderTarget::GLRenderTarget (std::string name) :
	RenderTarget (name, 0, 0),
	m_DepthBuffer (0),
//	m_RTCount (0),
	m_NoDrawAndRead (false),
	m_RenderTargets(0)
{
	m_Samples = 0;

	glGenFramebuffers (1, &m_Id);
}


GLRenderTarget::~GLRenderTarget(void) {

	glDeleteFramebuffers (1, &m_Id);
	glDeleteRenderbuffers(1, &m_DepthBuffer);
	if (m_Color)
		glDeleteRenderbuffers(m_Color, (GLuint *)&m_RenderTargets[0]);
}


void 
GLRenderTarget::init() {

	m_Init = true;
	m_RenderTargets.resize(IRenderer::MaxColorAttachments, -1);
	m_TexId.resize(IRenderer::MaxColorAttachments, NULL);
}


bool
GLRenderTarget::checkStatus() {

	glBindFramebuffer(GL_FRAMEBUFFER, m_Id);
	GLenum e = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	if (e == GL_FRAMEBUFFER_COMPLETE)
		return true;
	else
		return false;
}


void
GLRenderTarget::bind (void) {

	glBindFramebuffer (GL_FRAMEBUFFER, m_Id);
	if (m_Color)
		glDrawBuffers(m_Color, (GLenum *)&m_RenderTargets[0]);
}


void 
GLRenderTarget::unbind (void) {

	glBindFramebuffer (GL_FRAMEBUFFER, 0);
}


void 
GLRenderTarget::addColorTarget (std::string name, std::string internalFormat) {

	assert(m_Color < (unsigned int)IRenderer::MaxColorAttachments);

	if (m_Color >= (unsigned int)IRenderer::MaxColorAttachments)
		return;

	if (!m_Init)
		init();

	//if (m_Samples > 1)
	//	m_TexId[m_Color] = RESOURCEMANAGER->createTextureMS
	//		(name, internalFormat,m_Width, m_Height, m_Samples);
	//else
		m_TexId[m_Color] = RESOURCEMANAGER->createTexture
			(name, internalFormat,m_Width, m_Height, 1, m_Layers, 1, m_Samples);

	bind();
	attachColorTexture(m_TexId[m_Color], m_Color);
	setDrawBuffers();
	unbind();	

	m_Color++;

}


void 
GLRenderTarget::addDepthTarget (std::string name, std::string internalFormat) {

	if (!m_Init)
		init();

	m_Depth = 1;

	if (0 != m_DepthTexture) {
		RESOURCEMANAGER->removeTexture(m_DepthTexture->getLabel());
		m_DepthTexture = NULL;
	}

	//if (m_Samples > 1)
	//	m_DepthTexture = RESOURCEMANAGER->createTextureMS
	//		(name, internalFormat,m_Width, m_Height, m_Samples);
	//else
		m_DepthTexture = RESOURCEMANAGER->createTexture
			(name, internalFormat, m_Width, m_Height, 1, m_Layers, 1, m_Samples);

	bind();
	attachDepthStencilTexture(m_DepthTexture, GL_DEPTH_ATTACHMENT);
	unbind();
}


void 
GLRenderTarget::addStencilTarget (std::string name) {

	if (!m_Init)
		init();

	m_Stencil = 1;

	if (0 != m_StencilTexture) {
		RESOURCEMANAGER->removeTexture(m_StencilTexture->getLabel());
		m_StencilTexture = 0;
	}

	//if (m_Samples > 1)
	//	m_StencilTexture = RESOURCEMANAGER->createTextureMS
	//		(name, "STENCIL_INDEX8",m_Width, m_Height, m_Samples);
	//else
		m_StencilTexture = RESOURCEMANAGER->createTexture
			(name, "STENCIL_INDEX8", m_Width, m_Height, 1, m_Layers, 1, m_Samples);

	bind();
	attachDepthStencilTexture(m_StencilTexture, GL_STENCIL_ATTACHMENT);
	unbind();
}


void 
GLRenderTarget::addDepthStencilTarget (std::string name) {

	if (!m_Init)
		init();

	m_Depth = 1;
	m_Stencil = 1;

	if (0 != m_DepthTexture) {
		RESOURCEMANAGER->removeTexture(m_DepthTexture->getLabel());
		m_DepthTexture = 0;
	}

	//if (m_Samples > 1)
	//	m_DepthTexture = RESOURCEMANAGER->createTextureMS
	//		(name, "DEPTH24_STENCIL8",m_Width, m_Height, m_Samples);
	//else
		m_DepthTexture = RESOURCEMANAGER->createTexture
			(name, "DEPTH24_STENCIL8", m_Width, m_Height, 1, m_Layers, 1, m_Samples);

	bind();
	dettachDepthStencilTexture(GL_DEPTH_ATTACHMENT);
	attachDepthStencilTexture(m_DepthTexture, GL_DEPTH_STENCIL_ATTACHMENT);
	unbind();
}


void
GLRenderTarget::attachColorTexture (Texture* aTexture, unsigned int colorAttachment) {

	  if (-1 == (int) m_RenderTargets[colorAttachment]) {
			m_RenderTargets[m_Color] = GL_COLOR_ATTACHMENT0 + colorAttachment;
		}
		glFramebufferTexture  (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + colorAttachment, aTexture->getPropi(Texture::ID), 0);
}


void
GLRenderTarget::dettachColorTexture (unsigned int colorAttachment) {

	glFramebufferTexture  (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + colorAttachment, 0, 0);
	m_RenderTargets[(int)colorAttachment] = -1;
	--m_Color;
}


void 
GLRenderTarget::attachDepthStencilTexture (Texture* aTexture, GLuint type) {

	glFramebufferTexture (GL_FRAMEBUFFER, type, aTexture->getPropi(Texture::ID), 0);
}


void
GLRenderTarget::dettachDepthStencilTexture (GLuint type) {

	glFramebufferTexture (GL_FRAMEBUFFER, type, 0, 0);
}


void
GLRenderTarget::setDrawBuffers (void) {

	if (m_Color)
		glDrawBuffers(m_Color, (unsigned int *)&m_RenderTargets[0]);
}


void 
GLRenderTarget::_rebuild() {

	std::string texName;
	std::string internalFormat;

	if (m_Color > 0) {


		for (unsigned int i = 0; i < m_Color; ++i) {

			if (0 != m_TexId[i]) {
				texName = m_TexId[i]->getLabel();
				internalFormat = m_TexId[i]->Attribs.getListStringOp(Texture::INTERNAL_FORMAT, m_TexId[i]->getPrope(Texture::INTERNAL_FORMAT));;
				RESOURCEMANAGER->removeTexture (m_TexId[i]->getLabel());
				m_TexId[i] = NULL;
			}

			m_TexId[i] = RESOURCEMANAGER->createTexture 
				(texName, internalFormat, /*format, type, */m_Width, m_Height);

			bind();
			attachColorTexture (m_TexId[i], i);
			unbind();
		}
		setDrawBuffers();
	}  
	if (m_Depth) {

		if (0 != m_DepthTexture) {
			texName = m_DepthTexture->getLabel();
			internalFormat = m_DepthTexture->Attribs.getListStringOp(Texture::INTERNAL_FORMAT, m_DepthTexture->getPrope(Texture::INTERNAL_FORMAT));;
			RESOURCEMANAGER->removeTexture(m_DepthTexture->getLabel());
			m_DepthTexture = 0;
		}

		//if (m_Samples > 1)
		//	m_DepthTexture = RESOURCEMANAGER->createTextureMS
		//	(texName, "DEPTH24_STENCIL8", m_Width, m_Height, m_Samples);
		else
			m_DepthTexture = RESOURCEMANAGER->createTexture
			(texName, "DEPTH24_STENCIL8", m_Width, m_Height, 1, m_Layers, 1, m_Samples);

		bind();
		attachDepthStencilTexture(m_DepthTexture, GL_DEPTH_ATTACHMENT);
		unbind();
	}

	if (m_Stencil) {

		if (0 != m_StencilTexture) {
			texName = m_StencilTexture->getLabel();
			internalFormat = m_StencilTexture->Attribs.getListStringOp(Texture::INTERNAL_FORMAT, m_StencilTexture->getPrope(Texture::INTERNAL_FORMAT));;
			RESOURCEMANAGER->removeTexture(m_StencilTexture->getLabel());
			m_StencilTexture = 0;
		}

		//if (m_Samples > 1)
		//	m_StencilTexture = RESOURCEMANAGER->createTextureMS(texName, "STENCIL_INDEX8", m_Width, m_Height, m_Samples);
		//else
			m_StencilTexture = RESOURCEMANAGER->createTexture(texName, "STENCIL_INDEX8", 
						m_Width, m_Height, 1, m_Layers, 1, m_Samples);

		bind();
		attachDepthStencilTexture(m_StencilTexture, GL_DEPTH_ATTACHMENT);
		unbind();
	}

}

