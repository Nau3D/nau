#ifndef BUFFER_H
#define BUFFER_H

#include <string>
#include <math.h>

#include <nau/attribute.h>

#include <nau/config.h>

using namespace nau;


namespace nau
{
	namespace render
	{
		class Buffer
		{
		public:

			typedef enum { 
				COUNT_ENUMPROPERTY} EnumProperty;

			typedef enum { 
				COUNT_INTPROPERTY} IntProperty;

			typedef enum { ID, SIZE, COUNT_UINTPROPERTY} UIntProperty;

			typedef enum {COUNT_FLOAT4PROPERTY} Float4Property;
			typedef enum {COUNT_FLOATPROPERTY} FloatProperty;


			static AttribSet Attribs;

			std::map<int,int> m_IntProps;
			std::map<int,int> m_EnumProps;
			std::map<int,unsigned int> m_UIntProps;
			std::map<int,bool> m_BoolProps;
			std::map<int, vec4> m_Float4Props;
			std::map<int, float> m_FloatProps;

			// Note: no validation is performed!
			void setProp(int prop, Enums::DataType type, void *value);

			int getPropi(IntProperty prop);
			int getPrope(EnumProperty prop);
			unsigned int getPropui(UIntProperty prop);
			bool getPropb(BoolProperty prop);
			void *getProp(int prop, Enums::DataType type);

			void initArrays();

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
