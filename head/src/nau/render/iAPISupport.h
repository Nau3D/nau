#ifndef IAPISUPPORT_H
#define IAPISUPPORT_H

#include <map>
#include <string>



namespace nau
{
	namespace render
	{
		class IAPISupport
		{
		public:	

			enum APIFeatureSupport {
				OK, // this is required for attribute setting
				BUFFER_ATOMICS,
				BUFFER_SHADER_STORAGE,
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

			static IAPISupport *GetInstance();
			virtual void setAPISupport() = 0;
			bool apiSupport(APIFeatureSupport feature);
			unsigned int getVersion();

		protected:
			std::map<APIFeatureSupport, bool> m_APISupport;
			unsigned int m_Version;
			static IAPISupport *Instance;
			IAPISupport();
			~IAPISupport();
		};
	};
};

#endif
