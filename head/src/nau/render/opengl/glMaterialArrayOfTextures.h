#ifndef GL_MATERIAL_ARRAY__TEXTUREH
#define GL_MATERIAL_ARRAY_TEXTURE_H


#include "nau/material/iMaterialArrayOfTextures.h"

namespace nau {

	namespace render{

		class GLMaterialArrayOfTextures : public nau::material:: IMaterialArrayOfTextures
		{
		public:
			GLMaterialArrayOfTextures(void);
			~GLMaterialArrayOfTextures(void);

			void bind(void);
			void unbind(void);

		protected:
			static bool Init(void);
			static bool Inited;
		};
	};
};

#endif