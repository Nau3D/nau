#ifndef PASS_PRE_PROCESS_BUFFER_H
#define PASS_PRE_PROCESS_BUFFER_H

#include "nau/render/passProcessItem.h"
#include "nau/render/ibuffer.h"

namespace nau {
	namespace render {
	
		class PassProcessBuffer : public PassProcessItem {

		public:

			PassProcessBuffer();
			
			BOOL_PROP(CLEAR, 0);
			BOOL_PROP(MIPMAP, 1);

			INT_PROP(CLEAR_LEVEL, 0);

			static AttribSet Attribs;

			virtual void process();
			void setItem(IBuffer *buf);

		protected:

			IBuffer *m_Buffer;

			static bool Init();
			static bool Inited;
		};
	};
};


#endif
