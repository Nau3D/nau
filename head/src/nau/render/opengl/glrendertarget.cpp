#include <nau/render/opengl/glrendertarget.h>
#include <nau.h>

using namespace nau::render;


GLRenderTarget::GLRenderTarget(std::string name, unsigned int width, unsigned int height) :
	RenderTarget (name, width, height),
	
	m_DepthBuffer (0),
	m_RTCount (0),
	m_NoDrawAndRead (false)
{
	m_Samples = 0;
	for (int i = 0; i < RenderTarget::RENDERTARGETS; ++i) {
		m_RenderTargets[i] = -1;
	}

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
	m_RTCount (0),
	m_NoDrawAndRead (false)
{
	m_Samples = 0;
	for (int i = 0; i < RenderTarget::RENDERTARGETS; i++) {
		m_RenderTargets[i] = -1;
	}

	glGenFramebuffers (1, &m_Id);
}


GLRenderTarget::~GLRenderTarget(void)
{
	glDeleteFramebuffers (1, &m_Id);
	glDeleteRenderbuffers(1, &m_DepthBuffer);
	glDeleteRenderbuffers(m_RTCount, m_RenderTargets);
}



bool
GLRenderTarget::checkStatus() {

	GLenum e = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (e == GL_FRAMEBUFFER_COMPLETE)
		return true;
	else
		return false;
}

void
GLRenderTarget::bind (void)
{
	glBindFramebuffer (GL_FRAMEBUFFER, m_Id);
	glDrawBuffers(m_RTCount, m_RenderTargets);
}


void 
GLRenderTarget::unbind (void)
{
	glBindFramebuffer (GL_FRAMEBUFFER, 0);
}


void 
GLRenderTarget::addColorTarget (std::string name, std::string internalFormat)
{
	if (m_Color >= MAXFBOs) 
		return;

	if (m_Samples > 1)
		m_TexId[m_Color] = RESOURCEMANAGER->createTextureMS 
			(name, internalFormat,m_Width, m_Height, m_Samples);

	else
		m_TexId[m_Color] = RESOURCEMANAGER->createTexture 
			(name, internalFormat,m_Width, m_Height);

	bind();
	attachColorTexture (m_TexId[m_Color], (RenderTarget::ColorAttachment)m_Color);
	unbind();	

	m_Color++;

	setDrawBuffers();
}


//void 
//GLRenderTarget::addColorTarget (std::string name, std::string internalFormat, int samples)
//{
//	if (m_Color >= MAXFBOs) 
//		return;
//
//	m_TexId[m_Color] = RESOURCEMANAGER->createTextureMS(name, internalFormat,m_Width, m_Height, samples);
//
//	bind();
//	attachColorTexture (m_TexId[m_Color], (RenderTarget::ColorAttachment)m_Color);
//	unbind();	
//
//	m_Color++;
//
//	setDrawBuffers();
//}


void 
GLRenderTarget::addDepthTarget (std::string name, std::string internalFormat)
{
	m_Depth = 1;

	if (0 != m_TexId[MAXFBOs]) {
		RESOURCEMANAGER->removeTexture (m_TexId[MAXFBOs]->getLabel());
		m_TexId[MAXFBOs] = 0;			
	}

	if (m_Samples > 1)
		m_TexId[MAXFBOs] = RESOURCEMANAGER->createTextureMS 
			(name, internalFormat,m_Width, m_Height, m_Samples);
	else
		m_TexId[MAXFBOs] = RESOURCEMANAGER->createTexture 
			(name, internalFormat, m_Width, m_Height);

	bind();
	attachDepthStencilTexture (m_TexId[MAXFBOs], GL_DEPTH_ATTACHMENT);
	unbind();
}


void 
GLRenderTarget::addStencilTarget (std::string name)
{
	m_Stencil = 1;

	if (0 != m_TexId[MAXFBOs+1]) {
		RESOURCEMANAGER->removeTexture (m_TexId[MAXFBOs+1]->getLabel());
		m_TexId[MAXFBOs+1] = 0;			
	}

	if (m_Samples > 1)
		m_TexId[MAXFBOs+1] = RESOURCEMANAGER->createTextureMS 
			(name, "STENCIL_INDEX8",m_Width, m_Height, m_Samples);
	else
		m_TexId[MAXFBOs+1] = RESOURCEMANAGER->createTexture 
			(name, "STENCIL_INDEX8", m_Width, m_Height);

	bind();
	attachDepthStencilTexture (m_TexId[MAXFBOs+1], GL_STENCIL_ATTACHMENT);
	unbind();
}


void 
GLRenderTarget::addDepthStencilTarget (std::string name)
{
	m_Depth = 1;
	m_Stencil = 1;

	if (0 != m_TexId[MAXFBOs]) {
		RESOURCEMANAGER->removeTexture (m_TexId[MAXFBOs]->getLabel());
		m_TexId[MAXFBOs] = 0;			
	}

	if (m_Samples > 1)
		m_TexId[MAXFBOs] = RESOURCEMANAGER->createTextureMS 
			(name, "DEPTH24_STENCIL8",m_Width, m_Height, m_Samples);
	else
		m_TexId[MAXFBOs] = RESOURCEMANAGER->createTexture 
			(name, "DEPTH24_STENCIL8", m_Width, m_Height);

	bind();
	dettachDepthStencilTexture(GL_DEPTH_ATTACHMENT);
	attachDepthStencilTexture (m_TexId[MAXFBOs], GL_DEPTH_STENCIL_ATTACHMENT);
	unbind();
}


bool 
GLRenderTarget::attachColorTexture (Texture* aTexture, ColorAttachment colorAttachment)
{	
	if (m_RTCount < RenderTarget::RENDERTARGETS){
	  if (-1 == (int) m_RenderTargets[(int) colorAttachment]) {
			m_RenderTargets[m_RTCount] = GL_COLOR_ATTACHMENT0 + (int)colorAttachment;
			m_RTCount++;
		}
		glFramebufferTexture2D  (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + (int)colorAttachment, aTexture->getPrope(Texture::DIMENSION), aTexture->getPropui(Texture::ID), 0);
	}
	else {
		return false;
	}
	return true; 
}


bool
GLRenderTarget::dettachColorTexture (ColorAttachment colorAttachment)
{
   if ((int) m_RenderTargets[(int) colorAttachment] != -1) {
		glFramebufferTexture2D  (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + (int)colorAttachment, GL_TEXTURE_2D, 0, 0);
		m_RenderTargets[(int)colorAttachment] = -1;
		--m_RTCount;
	}
	return true;
}


bool 
GLRenderTarget::attachDepthStencilTexture (Texture* aTexture, GLuint type)
{
	glFramebufferTexture2D (GL_FRAMEBUFFER, type, aTexture->getPrope(Texture::DIMENSION), aTexture->getPropui(Texture::ID), 0);
	//CHECK_FRAMEBUFFER_STATUS();

	return true;
}


bool
GLRenderTarget::dettachDepthStencilTexture (GLuint type)
{
	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
	glFramebufferTexture2D (GL_FRAMEBUFFER, type, GL_TEXTURE_2D, 0, 0);

	return true;
}


void
GLRenderTarget::setDrawBuffers (void)
{
	glDrawBuffers(m_RTCount, m_RenderTargets);
}


void 
GLRenderTarget::_rebuild() 
{
	if (m_Color > 0) {

		std::string texName;
		std::string internalFormat;
		std::string format;
		std::string type;

		for (unsigned int i = 0; i < m_Color; ++i) {

			if (0 != m_TexId[i]) {
				texName = m_TexId[i]->getLabel();
				internalFormat = m_TexId[i]->Attribs.getListStringOp(Texture::INTERNAL_FORMAT, m_TexId[MAXFBOs]->getPrope(Texture::INTERNAL_FORMAT));;
				format = m_TexId[i]->Attribs.getListStringOp(Texture::FORMAT, m_TexId[MAXFBOs]->getPrope(Texture::FORMAT));
				type = m_TexId[i]->Attribs.getListStringOp(Texture::TYPE, m_TexId[MAXFBOs]->getPrope(Texture::TYPE));
				RESOURCEMANAGER->removeTexture (m_TexId[i]->getLabel());
				m_TexId[i] = 0;
			}

			m_TexId[i] = RESOURCEMANAGER->createTexture 
				(texName, internalFormat, format, type, m_Width, m_Height);

			bind();
			attachColorTexture (m_TexId[i], (RenderTarget::ColorAttachment)i);
			unbind();
		}
		setDrawBuffers();
	}  
	if (m_Depth) {

		std::string texName;
		std::string internalFormat;
		std::string format;
		std::string type;

		if (0 != m_TexId[MAXFBOs]) {
			texName = m_TexId[MAXFBOs]->getLabel();
			internalFormat = m_TexId[MAXFBOs]->Attribs.getListStringOp(Texture::INTERNAL_FORMAT, m_TexId[MAXFBOs]->getPrope(Texture::INTERNAL_FORMAT));;
			format = m_TexId[MAXFBOs]->Attribs.getListStringOp(Texture::FORMAT, m_TexId[MAXFBOs]->getPrope(Texture::FORMAT));
			type = m_TexId[MAXFBOs]->Attribs.getListStringOp(Texture::TYPE, m_TexId[MAXFBOs]->getPrope(Texture::TYPE));
			RESOURCEMANAGER->removeTexture (m_TexId[MAXFBOs]->getLabel());
			m_TexId[MAXFBOs] = 0;			
		}

		m_TexId[MAXFBOs] = RESOURCEMANAGER->createTexture 
			(texName, internalFormat, format, type, m_Width, m_Height);

		bind();
		attachDepthStencilTexture (m_TexId[MAXFBOs], GL_DEPTH_ATTACHMENT);
		unbind();
	}
}

