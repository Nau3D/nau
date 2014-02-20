#ifndef GLRENDERTARGET_H
#define GLRENDERTARGET_H

#include <assert.h>

#include <nau/render/rendertarget.h>

#include <GL/glew.h>

#define CHECK_FRAMEBUFFER_STATUS() \
{ \
	GLenum status; \
	status = glCheckFramebufferStatus (GL_FRAMEBUFFER); \
	switch (status) { \
		case GL_FRAMEBUFFER_COMPLETE: \
			break; \
		case GL_FRAMEBUFFER_UNSUPPORTED: \
			break; \
		default: \
			assert (0); \
	} \
} \

namespace nau
{
	namespace render
	{

		class GLRenderTarget : public RenderTarget
		{
		friend class RenderTarget;

		private:
			//unsigned int m_Depth; 
			//unsigned int m_Color; // number of color render targets

		public:
			virtual void bind (void);			
			virtual void unbind (void);

			virtual bool checkStatus();

			void addColorTarget (std::string name, std::string internalFormat);
			void addDepthTarget (std::string name, std::string internalFormat);
			void addStencilTarget (std::string name);
			void addDepthStencilTarget(std::string name);
			~GLRenderTarget(void);

		protected:
			virtual bool attachDepthStencilTexture (Texture* aTexture, GLuint type);
			virtual bool dettachDepthStencilTexture (GLuint type);
			virtual bool attachColorTexture (nau::render::Texture* aTexture, ColorAttachment colorAttachment); 
			virtual bool dettachColorTexture (ColorAttachment colorAttachment);

			void setDrawBuffers (void);

			GLuint m_DepthBuffer;
			GLenum m_RenderTargets[RenderTarget::RENDERTARGETS];
			int m_RTCount;
			bool m_NoDrawAndRead;

			GLRenderTarget (std::string name, unsigned int width, unsigned int height);
			GLRenderTarget (std::string name);

			void _rebuild();
		};
	};
};


#endif
