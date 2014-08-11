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

		protected:
			unsigned int m_Id; 
			unsigned int m_Color; // number of color targets;
			unsigned int m_Depth;
			unsigned int m_Stencil;
			unsigned int m_Samples;
			unsigned int m_Layers;
			unsigned int m_Width;
			unsigned int m_Height;
			std::string m_Name;
			std::vector<nau::render::Texture*> m_TexId;
			Texture *m_DepthTexture;
			Texture *m_StencilTexture;

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
			void setLayerCount(int layers);
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
