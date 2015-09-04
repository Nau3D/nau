#include "nau/config.h"


#ifndef IBUFFER_H
#define IBUFFER_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <string>



using namespace nau;


namespace nau
{
	namespace material
	{
		class IBuffer : public AttributeValues
		{
		public:

			INT_PROP(ID,0);

			UINT_PROP(SIZE, 0);
			UINT_PROP(STRUCT_SIZE, 1);

			ENUM_PROP(CLEAR, 0);

			UINT3_PROP(DIM, 0);

			typedef enum CV{
				NEVER,
				BY_FRAME
			} ClearValues; 

			static AttribSet Attribs;


			// Note: no validation is performed!
			virtual void setPropui(UIntProperty  prop, unsigned int value) = 0;
			virtual void setPropui3(UInt3Property  prop, uivec3 &v) = 0;

			static IBuffer* Create(std::string label);

			std::string& getLabel (void);

			virtual void setData(unsigned int size, void *data) = 0;
			virtual void setSubData(unsigned int offset, unsigned int size, void*data) = 0;
			virtual void setSubDataNoBinding(unsigned int bufferType, unsigned int offset, unsigned int size, void*data) = 0;
			// returns the number of bytes read
			virtual int getData(unsigned int offset, unsigned int size, void *data) = 0;

			virtual void bind(unsigned int type) = 0;
			virtual void unbind() =0;
			virtual void clear() = 0;

			virtual IBuffer * clone() = 0;

			// Only useful for GUIs
			void setStructure(std::vector<Enums::DataType>);
			std::vector<Enums::DataType> &getStructure();

			virtual void refreshBufferParameters() = 0;

			void appendItemToStruct(Enums::DataType);
		
			~IBuffer(void) {};
		
		protected:

			IBuffer() : m_Label("") { registerAndInitArrays(Attribs); };

			static bool Init();
			static bool Inited;

			std::string m_Label;
			std::vector<Enums::DataType> m_Structure;
		};
	};
};



#endif // IBUFFER_H