#ifndef MATERIALBUFFER_H
#define MATERIALBUFFER_H

#include <nau/attribute.h>
#include <nau/attributeValues.h>
#include <nau/render/ibuffer.h>


#include <string>
#include <math.h>

using namespace nau;


namespace nau
{
	namespace material
	{
		class IMaterialBuffer : public AttributeValues
		{
		public:

			INT_PROP(BINDING_POINT, 0);
			ENUM_PROP(TYPE, 0);
			BOOL_PROP(CLEAR, 0);

			static AttribSet Attribs;

			static IMaterialBuffer* Create(nau::render::IBuffer *b);

			std::string& getLabel(void) {
				return m_Label;
			};

			virtual void bind() = 0;
			virtual void unbind() = 0;
			void setBuffer(nau::render::IBuffer *b);

			~IMaterialBuffer(void) {};

		protected:

			IMaterialBuffer() { registerAndInitArrays("MATERIAL_BUFFER", Attribs); };

			static bool Init();
			static bool Inited;

			std::string m_Label;
			nau::render::IBuffer *m_Buffer;
		};
	};
};


#endif // IBUFFER_H