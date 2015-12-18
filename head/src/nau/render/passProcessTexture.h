#ifndef PASS_PRE_PROCESS_TEXTURE_H
#define PASS_PRE_PROCESS_TEXTURE_H

#include "nau/material/iTexture.h"
#include "nau/render/passProcessItem.h"

using namespace nau::material;

namespace nau {
	namespace render {
	
		class PassProcessTexture : public PassProcessItem {

		public:


			PassProcessTexture();
			
			BOOL_PROP(CLEAR, 0);
			BOOL_PROP(MIPMAP, 1);

			INT_PROP(CLEAR_LEVEL, 0);

			static AttribSet Attribs;

			virtual void process();
			void setItem(ITexture *tex);

		protected:

			ITexture *m_Tex;

			static bool Init();
			static bool Inited;
		};
	};
};


#endif

