#ifndef BUFFER_H
#define BUFFER_H

#include <string>
#include <math.h>

#include <nau/attribute.h>
#include <nau/attributeValues.h>

#include <nau/config.h>

using namespace nau;


namespace nau
{
	namespace render
	{
		class Buffer : public AttributeValues
		{
		public:

			INT_PROP(ID,0);
			INT_PROP(BINDING_POINT, 1);

			UINT(SIZE, 0);

			static AttribSet Attribs;


			// Note: no validation is performed!
			void setProp(int prop, Enums::DataType type, void *value);

			void *getProp(int prop, Enums::DataType type);

			static Buffer* Create (std::string label, int size);

			virtual std::string& getLabel (void);
			virtual void setLabel (std::string label);

			virtual void bind() = 0;
		
			virtual ~Buffer(void);

		protected:
			static bool Init();
			static bool Inited;

			std::string m_Label;

		};
	};
};

#endif
