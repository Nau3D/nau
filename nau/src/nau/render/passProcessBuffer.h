#ifndef PASS_PRE_PROCESS_BUFFER_H
#define PASS_PRE_PROCESS_BUFFER_H

#include "nau/material/iBuffer.h"
#include "nau/render/passProcessItem.h"

namespace nau {
	namespace render {
	
		class PassProcessBuffer : public PassProcessItem {

		public:

			PassProcessBuffer();
			
			BOOL_PROP(CLEAR, 0);

			static AttribSet Attribs;
			static AttribSet &GetAttribs() { return Attribs; }

			virtual void process();
			void setItem(nau::material::IBuffer *buf);

		protected:

			nau::material::IBuffer *m_Buffer;

			static bool Init();
			static bool Inited;
		};
	};
};


#endif

