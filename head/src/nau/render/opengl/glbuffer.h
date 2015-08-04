#include "nau/config.h"

#ifndef GLBUFFER_H
#define GLBUFFER_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/config.h"
#include "nau/render/iBuffer.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <string>

using namespace nau;


namespace nau
{
	namespace render
	{
		class GLBuffer : public IBuffer
		{
		public:

			GLBuffer(std::string label);
			~GLBuffer(void) ;

			void bind(unsigned int target);
			void unbind();
			void setPropui(UIntProperty  prop, unsigned int value);
			void setPropui3(UInt3Property  prop, uivec3 &v);
			void setData(unsigned int size, void *data);
			void setSubData(unsigned int offset, unsigned int size, void*data);
			int getData(unsigned int offset, unsigned int size, void *data);
			void clear();
			IBuffer * clone();

			//! Should be called before getting the size
			// and other properties
			void refreshBufferParameters();

		protected:
			static bool Init();
			static bool Inited;

			GLBuffer() {};

			int m_LastBound;

		};
	};
};



#endif // GLBUFFER_H