#ifndef PASS_PRE_PROCESS_TEXTURE_H
#define PASS_PRE_PROCESS_TEXTURE_H

#include "nau/render/passProcessItem.h"
#include "nau/render/texture.h"

namespace nau {
	namespace render {
	
		class PassProcessTexture : public PassProcessItem {

		public:

			PassProcessTexture();
			
			BOOL_PROP(CLEAR, 0);
			BOOL_PROP(MIPMAP, 1);

			INT_PROP(CLEAR_LEVEL, 0);

			virtual void process();
			void setItem(Texture *tex);

		protected:

			Texture *m_Tex;

			static bool Init();
			static bool Inited;
		};
	};
};


#endif

