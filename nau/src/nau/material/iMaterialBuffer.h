#ifndef MATERIAL_BUFFER_H
#define MATERIAL_BUFFER_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/material/iBuffer.h"

#include <string>

using namespace nau;

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif


namespace nau
{
	namespace material
	{
		class IMaterialBuffer : public AttributeValues
		{
		public:

			INT_PROP(BINDING_POINT, 0);
			ENUM_PROP(TYPE, 0);

			static nau_API AttribSet Attribs;
			static nau_API AttribSet &GetAttribs();

			static nau_API IMaterialBuffer* Create(IBuffer *b);

			nau_API std::string& getLabel(void) {
				return m_Label;
			};

			nau_API virtual void bind() = 0;
			nau_API virtual void unbind() = 0;
			nau_API void setBuffer(IBuffer *b);
			nau_API IBuffer *getBuffer();

			nau_API virtual ~IMaterialBuffer(void) {};

		protected:

			IMaterialBuffer() { registerAndInitArrays(Attribs); };

			static bool Init();
			static bool Inited;

			std::string m_Label;
			IBuffer *m_Buffer;
		};
	};
};


#endif // IBUFFER_H