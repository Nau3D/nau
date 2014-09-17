#include <nau/config.h>

#if NAU_OPENGL_VERSION >= 430

#ifndef GLBUFFER_H
#define GLBUFFER_H

#include <string>
#include <math.h>

#include <nau/attribute.h>
#include <nau/attributeValues.h>

#include <nau/config.h>
#include <nau/render/ibuffer.h>

using namespace nau;


namespace nau
{
	namespace render
	{
		class GLBuffer : public IBuffer
		{
		public:

			GLBuffer(std::string label, int size);
			~GLBuffer(void) ;

			void bind();
			void unbind();
			void setProp(int prop, Enums::DataType type, void *value);
			void clear();
			IBuffer * clone();

		protected:
			static bool Init();
			static bool Inited;

			GLBuffer() {};

		};
	};
};

#endif // NAU_OPENGL_VERSION

#endif // GLBUFFER_H