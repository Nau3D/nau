#ifndef GLRENDERTARGET_H
#define GLRENDERTARGET_H

#include "nau/render/rendertarget.h"

#include <GL/glew.h>

#include <assert.h>

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
			void init();

			virtual void bind (void);			
			virtual void unbind (void);

			virtual bool checkStatus();

			void addColorTarget (std::string name, std::string internalFormat);
			void addDepthTarget (std::string name, std::string internalFormat);
			void addStencilTarget (std::string name);
			void addDepthStencilTarget(std::string name);
			~GLRenderTarget(void);

		protected:
			void attachDepthStencilTexture (Texture* aTexture, GLuint type);
			void dettachDepthStencilTexture(GLuint type);
			void attachColorTexture(nau::render::Texture* aTexture, unsigned int colorAttachment);
			void dettachColorTexture(unsigned int  colorAttachment);

			void setDrawBuffers (void);

			GLuint m_DepthBuffer;
			std::vector<int> m_RenderTargets;
			//int m_RTCount;
			bool m_NoDrawAndRead;

			GLRenderTarget (std::string name, unsigned int width, unsigned int height);
			GLRenderTarget (std::string name);

			void _rebuild();
			bool m_Init = false;
		};
	};
};


#endif
