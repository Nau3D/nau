#ifndef IAPISUPPORT_H
#define IAPISUPPORT_H

#include <map>
#include <string>

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
		class IAPISupport
		{
		public:	

			enum APIFeatureSupport {
				OK, // this is required for attribute setting
				BINDLESS_TEXTURES,
				BUFFER_ATOMICS,
				BUFFER_SHADER_STORAGE,
				BUFFER_UNIFORM,
				CLEAR_BUFFER,
				CLEAR_TEXTURE,
				CLEAR_TEXTURE_LEVEL,
				COMPUTE_SHADER,
				GEOMETRY_SHADER,
				IMAGE_TEXTURE,
				OBJECT_LABELS,
				RESET_TEXTURES,
				TESSELATION_SHADERS,
				TEX_STORAGE,
				TEXTURE_SAMPLERS,
				COUNT_API_SUPPORT
			} ;

			static nau_API IAPISupport *GetInstance();
			virtual void setAPISupport() = 0;
			nau_API bool apiSupport(APIFeatureSupport feature);
			nau_API unsigned int getVersion();
			~IAPISupport();

		protected:
			std::map<APIFeatureSupport, bool> m_APISupport;
			unsigned int m_Version;
			static IAPISupport *Instance;
			IAPISupport();
		};
	};
};

#endif
