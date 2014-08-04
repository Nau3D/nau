#ifndef IPROGRAMVALUE_H
#define IPROGRAMVALUE_H

#include <string>
#include <nau/enums.h>
#include <nau/attribute.h>

#include <nau/config.h>

namespace nau {

	namespace render {
      
		class IProgramValue {
	 
		public:

			typedef enum {SEMANTICS, COUNT_ENUMPROPERTY} EnumProperty;
			typedef enum {PROPERTY, SIZE, CARDINALITY, ID, COUNT_INTPROPERTY} IntProperty;
			typedef enum {CURRENT, COUNT_BOOLPROPERTY} BoolProperty;
 
			enum Semantic_Type {
				CAMERA,
				LIGHT,
				TEXTURE,
				IMAGE_TEXTURE,
				PASS,
				COLOR,
				DATA
			};

			enum Semantic_ValueOf {
				UNIT=100, 
				COUNT,
				USERDATA
			};

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
			bool getPropb(BoolProperty prop);

			void initArrays();

			/// Called by project loader
			static bool Validate(std::string type,std::string context,std::string component);

			/// This function is to be called when a program is linked
			static IProgramValue *Create(unsigned int programID, std::string name, std::string type, std::string context, std::string component, int id);

			void set(std::string name, std::string type, std::string context, std::string component, int id);
		protected:
			/// the name of the item where the variable is retrieved or "CURRENT"
			std::string m_Context;
			///
			int m_Component;
			/// the name of the variable
			std::string m_Name;
			/// the value of the variable
			void *m_Value = NULL;
			/// the position in the array of variables
			int m_Id;
			/// the type of the variable
			nau::Enums::DataType m_ValueType;
			/// the number of elements in the variable
			int m_Cardinality;
			/// the type of the variable
			Semantic_Type m_Type;

			IProgramValue();
			~IProgramValue();

			virtual void clone(IProgramValue *pv) = 0;

			std::string getName();

			/// get values from each variable from the respective item
			void* setValues ();

			void *allocate(nau::Enums::DataType dt, int cardinality);

			static bool Init();
			static bool Inited;

		};
	};
};
#endif //PROGRAMVALUE_H
