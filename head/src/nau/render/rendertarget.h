#ifndef RENDERTARGET_H
#define RENDERTARGET_H

#include <nau/render/rendertarget.h>
#include <nau/render/texture.h>

namespace nau
{
	namespace render
	{
		class RenderTarget
		{
		public:
			static const int RENDERTARGETS = 16;
			static const int MAXFBOs = 8;

			enum ColorAttachment {
				COLOR0,
				COLOR1,
				COLOR2,
				COLOR3,
				COLOR4,
				COLOR5,
				COLOR6,
				COLOR7,
				COLOR8,
				COLOR9,
				COLOR10,
				COLOR11,
				COLOR12,
				COLOR13,
				COLOR14,
				COLOR15
			};

		protected:
			unsigned int m_Id; 
			unsigned int m_Color; // number of color targets;
			unsigned int m_Depth;
			unsigned int m_Stencil;
			unsigned int m_Samples;
			unsigned int m_Width;
			unsigned int m_Height;
			std::string m_Name;
			nau::render::Texture* m_TexId[MAXFBOs+2];

			// clear values per channel
			nau::math::vec4 m_ClearValues;


		public:
			static RenderTarget* Create (std::string name, unsigned int width, unsigned int height);
			static RenderTarget* Create (std::string name);

			virtual bool checkStatus() = 0;

			virtual void bind (void) = 0;
			virtual void unbind (void) = 0;

			virtual void addColorTarget (std::string name, std::string internalFormat) = 0;
			virtual void addDepthTarget (std::string name, std::string internalFormat) = 0;
			virtual void addStencilTarget (std::string name) = 0;
			virtual void addDepthStencilTarget(std::string name) = 0;

			nau::render::Texture* getTexture(unsigned int i);

			void setClearValues(float r, float g, float b, float a);
			void setSampleCount(int samples);
			const nau::math::vec4 & getClearValues(); 
			virtual unsigned int getNumberOfColorTargets();
			virtual int getId (void);
			virtual std::string &getName (void);
			unsigned int getWidth (void);
			unsigned int getHeight (void);

			virtual ~RenderTarget(void) {};

		protected:
			RenderTarget (): m_Samples(0) {};
			RenderTarget (std::string name, unsigned int width, unsigned int height);

		};
	};
};

#endif 
