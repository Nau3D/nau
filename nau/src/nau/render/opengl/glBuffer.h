#include "nau/config.h"

#ifndef GLBUFFER_H
#define GLBUFFER_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/config.h"
#include "nau/material/iBuffer.h"

#include <glbinding/gl/gl.h>
using namespace gl;

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <string>

using namespace nau;
using namespace nau::material;


namespace nau
{
	namespace render
	{
		class GLBuffer : public IBuffer
		{
		public:

			friend class nau::material::IBuffer;

			static std::map<GLenum, GLenum> BufferBound;

			~GLBuffer(void) ;

			void bind(unsigned int target);
			void unbind();
			void setPropui(UIntProperty  prop, unsigned int value);
			void setPropui3(UInt3Property  prop, uivec3 &v);
			void setData(size_t size, void *data);
			void setSubData(size_t offset, size_t size, void*data);
			void setSubDataNoBinding(unsigned int bufferType, size_t offset, size_t size, void*data);
			size_t getData(size_t offset, size_t size, void *data);
			void clear();
			IBuffer * clone();

			//! Should be called before getting the size
			// and other properties
			void refreshBufferParameters();

		protected:
			static bool Init();
			static bool Inited;

			GLBuffer(std::string label);
			GLBuffer() {};
			
			char *m_BufferMapPointer;

			int m_LastBound;

		};
	};
};



#endif // GLBUFFER_H