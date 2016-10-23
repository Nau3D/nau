#ifndef GL_MATERIAL_BUFFER_H
#define GL_MATERIAL_BUFFER_H


#include "nau/material/iMaterialBuffer.h"

namespace nau {

	namespace render{

		class GLMaterialBuffer : public nau::material:: IMaterialBuffer
		{
		public:
			GLMaterialBuffer(void);
			~GLMaterialBuffer(void);

			void bind(void);
			void unbind(void);

			//void setProp(int prop, Enums::DataType type, void *value);

		protected:
			static bool Init(void);
			static bool Inited;
		};
	};
};

#endif