#include "nau/render/opengl/glrendertarget.h"
#include "nau.h"

#include <assert.h>

using namespace nau::render;


//GLRenderTarget::GLRenderTarget(std::string name, unsigned int width, unsigned int height) :
//	RenderTarget (name, width, height),
//	
//	m_DepthBuffer (0),
////	m_RTCount (0),
//	m_NoDrawAndRead (false),
//	m_RenderTargets(0)
//{
//	m_Samples = 0;
//	//for (int i = 0; i < RenderTarget::MAXFBOs; ++i) {
//	//	m_RenderTargets[i] = -1;
//	//}
//
//	glGenFramebuffers (1, &m_Id);
//	glGenRenderbuffers (1, &m_DepthBuffer);
//	glBindRenderbuffer (GL_RENDERBUFFER, m_DepthBuffer);
//	glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_Width, m_Height);
//
//	bind();
//	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_DepthBuffer);
//	unbind();
//
//	glBindRenderbuffer (GL_RENDERBUFFER, 0);
//}


GLRenderTarget::GLRenderTarget (std::string name) :
	RenderTarget (),
//	m_DepthBuffer (0),
//	m_RTCount (0),
//	m_NoDrawAndRead (false),
	m_RenderTargets(0)
{
	m_Name = name;
	m_DepthTexture = NULL;
	m_DepthBuffer = 0;
	m_Depth = 0;

	glGenFramebuffers (1, &m_Id);
}


GLRenderTarget::~GLRenderTarget(void) {

	glDeleteFramebuffers (1, &m_Id);
	//glDeleteRenderbuffers(1, &m_DepthBuffer);
	//if (m_Color)
	//	glDeleteRenderbuffers(m_Color, (GLuint *)&m_RenderTargets[0]);
}


void 
GLRenderTarget::init() {

	m_Init = true;
	m_RenderTargets.resize(IRenderer::MaxColorAttachments, -1);
	m_TexId.resize(IRenderer::MaxColorAttachments, NULL);

	glGenRenderbuffers (1, &m_DepthBuffer);
	glBindRenderbuffer (GL_RENDERBUFFER, m_DepthBuffer);
	glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_UInt2Props[SIZE].x,m_UInt2Props[SIZE].y);

	bind();
	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_DepthBuffer);
	unbind();

	glBindRenderbuffer (GL_RENDERBUFFER, 0);
}


void 
GLRenderTarget::setPropui2(UInt2Property prop, uivec2 &value) {

	switch (prop) {

	case RenderTarget::SIZE:
		if (m_UInt2Props[SIZE] != value) {
			m_UInt2Props[SIZE] = value;
			resize();
		}
		break;

	default: AttributeValues::setPropui2(prop, value);
	}
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

	m_TexId[m_Color] = RESOURCEMANAGER->createTexture
		(name, internalFormat,m_UInt2Props[SIZE].x,m_UInt2Props[SIZE].y, 1, 
		m_UIntProps[LAYERS], 1, m_UIntProps[SAMPLES]);

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

	m_DepthTexture = RESOURCEMANAGER->createTexture
		(name, internalFormat, m_UInt2Props[SIZE].x,m_UInt2Props[SIZE].y, 1, 
		m_UIntProps[LAYERS], 1, m_UIntProps[SAMPLES]);

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

	m_StencilTexture = RESOURCEMANAGER->createTexture
		(name, "STENCIL_INDEX8", m_UInt2Props[SIZE].x,m_UInt2Props[SIZE].y, 1, 
		m_UIntProps[LAYERS], 1, m_UIntProps[SAMPLES]);

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

	m_DepthTexture = RESOURCEMANAGER->createTexture
		(name, "DEPTH24_STENCIL8", m_UInt2Props[SIZE].x,m_UInt2Props[SIZE].y, 1, 
		m_UIntProps[LAYERS], 1, m_UIntProps[SAMPLES]);

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
GLRenderTarget::resize() {

	std::string texName;
	std::string internalFormat;

	bind();
	if (m_Color > 0) {

		for (unsigned int i = 0; i < m_Color; ++i) {

			if (0 != m_TexId[i]) {
				m_TexId[i]->resize(m_UInt2Props[SIZE].x, m_UInt2Props[SIZE].y, 1);
				//texName = m_TexId[i]->getLabel();
				//internalFormat = m_TexId[i]->Attribs.getListStringOp(Texture::INTERNAL_FORMAT, m_TexId[i]->getPrope(Texture::INTERNAL_FORMAT));;
				//RESOURCEMANAGER->removeTexture (m_TexId[i]->getLabel());
				//m_TexId[i] = NULL;
				//attachColorTexture (m_TexId[i], i);
			}

			//m_TexId[i] = RESOURCEMANAGER->createTexture 
			//	(texName, internalFormat, m_UInt2Props[SIZE].x,m_UInt2Props[SIZE].y, 1, 
			//	m_UIntProps[LAYERS], 1, m_UIntProps[SAMPLES]);

		}
		//setDrawBuffers();
	}  
	if (m_Depth) {

		if (0 != m_DepthTexture) {
				m_DepthTexture->resize(m_UInt2Props[SIZE].x, m_UInt2Props[SIZE].y, 1);
			//texName = m_DepthTexture->getLabel();
			//internalFormat = m_DepthTexture->Attribs.getListStringOp(Texture::INTERNAL_FORMAT, m_DepthTexture->getPrope(Texture::INTERNAL_FORMAT));;
			//RESOURCEMANAGER->removeTexture(m_DepthTexture->getLabel());
			//m_DepthTexture = 0;
		}

		//m_DepthTexture = RESOURCEMANAGER->createTexture
		//		(texName, "DEPTH24_STENCIL8", m_UInt2Props[SIZE].x,m_UInt2Props[SIZE].y, 1, 
		//		m_UIntProps[LAYERS], 1, m_UIntProps[SAMPLES]);

		//bind();
		//attachDepthStencilTexture(m_DepthTexture, GL_DEPTH_ATTACHMENT);
		//unbind();
	}

	if (m_Stencil) {

		if (0 != m_StencilTexture) {
			m_StencilTexture->resize(m_UInt2Props[SIZE].x, m_UInt2Props[SIZE].y, 1);
			//texName = m_StencilTexture->getLabel();
			//internalFormat = m_StencilTexture->Attribs.getListStringOp(Texture::INTERNAL_FORMAT, m_StencilTexture->getPrope(Texture::INTERNAL_FORMAT));;
			//RESOURCEMANAGER->removeTexture(m_StencilTexture->getLabel(//*));
			//m_StencilTexture = 0;
		}

		//m_StencilTexture = RESOURCEMANAGER->createTexture
		//	(texName, "STENCIL_INDEX8", m_UInt2Props[SIZE].x,m_UInt2Props[SIZE].y, 1, 
		//	m_UIntProps[LAYERS], 1, m_UIntProps[SAMPLES]);

		//bind();
		//attachDepthStencilTexture(m_StencilTexture, GL_DEPTH_ATTACHMENT);
		//unbind();
	}
	unbind();

}

