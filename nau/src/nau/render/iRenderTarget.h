#ifndef RENDERTARGET_H
#define RENDERTARGET_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/material/iTexture.h"
#include "nau/render/iRenderTarget.h"

#include <string>
#include <vector>

using namespace nau::material;

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

namespace nau
{
	namespace render
	{
		class IRenderTarget: public AttributeValues
		{
		public:

			UINT_PROP(SAMPLES, 0);
			UINT_PROP(LAYERS, 1);
			UINT_PROP(LEVELS, 2);

			UINT2_PROP(SIZE, 0);

			FLOAT4_PROP(CLEAR_VALUES, 0);

			static AttribSet Attribs;
			static nau_API AttribSet &GetAttribs();

		protected:
			unsigned int m_Id; 
			unsigned int m_Color; // number of color targets;
			bool m_CubeMap;
			unsigned int m_Depth;
			unsigned int m_Stencil;
			//unsigned int m_Samples;
			//unsigned int m_Layers;
			//unsigned int m_Width;
			//unsigned int m_Height;
			std::string m_Name;
			std::vector<ITexture*> m_TexId;
			ITexture *m_DepthTexture;
			ITexture *m_StencilTexture;

			// clear values per channel
			//nau::math::vec4 m_ClearValues;


		public:

			//static IRenderTarget* nau_API Create (std::string name, unsigned int width, unsigned int height);
			static nau_API IRenderTarget* Create (std::string name);

			nau_API virtual bool checkStatus() = 0;
			nau_API virtual void getErrorMessage(std::string &message) = 0;
			nau_API virtual void resize() = 0;

			virtual void bind (void) = 0;
			virtual void unbind (void) = 0;

			nau_API virtual void addColorTarget(std::string name, std::string internalFormat) = 0;
			nau_API virtual void addCubeMapTarget(std::string name, std::string internalFormat) = 0;
			nau_API virtual void addDepthTarget(std::string name, std::string internalFormat) = 0;
			nau_API virtual void addStencilTarget (std::string name) = 0;
			nau_API virtual void addDepthStencilTarget(std::string name) = 0;

			nau_API nau::material::ITexture* getTexture(unsigned int i);
			nau_API nau::material::ITexture* getDepthTexture();
			nau_API nau::material::ITexture* getStencilTexture();

			nau_API virtual void setPropui2(UInt2Property prop, uivec2 &value) = 0;

			nau_API virtual unsigned int getNumberOfColorTargets();
			nau_API virtual int getId (void);
			nau_API virtual std::string &getName (void);

			nau_API virtual ~IRenderTarget(void) {};

		protected:
			IRenderTarget();
			static bool Init();
			static bool Inited;

		};
	};
};

#endif 
