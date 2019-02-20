#ifndef GLRENDERTARGET_H
#define GLRENDERTARGET_H

#include "nau/render/iRenderTarget.h"

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>

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

		class GLRenderTarget : public IRenderTarget
		{
		friend class IRenderTarget;

		public:
			void init();

			void bind (void);			
			void unbind (void);

			bool checkStatus();
			void getErrorMessage(std::string &message);
			void resize();

			void addColorTarget (std::string name, std::string internalFormat);
			void addCubeMapTarget(std::string name, std::string internalFormat);
			void addDepthTarget (std::string name, std::string internalFormat);
			void addStencilTarget (std::string name);
			void addDepthStencilTarget(std::string name);

			void setPropui2(UInt2Property prop, uivec2 &value);

			~GLRenderTarget(void);

		protected:
			//GLRenderTarget (std::string name, unsigned int width, unsigned int height);
			GLRenderTarget (std::string name);

			void attachDepthStencilTexture (ITexture* aTexture, GLuint type);
			void dettachDepthStencilTexture(GLuint type);
			void attachColorTexture(ITexture* aTexture, unsigned int colorAttachment);
			void dettachColorTexture(unsigned int  colorAttachment);

			void setDrawBuffers (void);
			bool m_Init = false;

			GLuint m_DepthBuffer;
			std::vector<int> m_RenderTargets;
			//int m_RTCount;
			//bool m_NoDrawAndRead;
		};
	};
};


#endif
