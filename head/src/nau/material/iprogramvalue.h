#ifndef PROGRAMVALUE_H
#define PROGRAMVALUE_H

#include <string>
#include "nau/enums.h"

namespace nau {

	namespace material {
      
		class ProgramValue {
	 
		public:
 
			enum SEMANTIC_TYPE {
				CAMERA,
				LIGHT,
				TEXTURE,
				DATA,
				PASS,
				CURRENT
			};

			enum SEMANTIC_VALUEOF {
			//	ID=100,
				UNIT=100, 
				COUNT,
			//	TYPE,
			//	FLOATS,// place int values before this point SEE dlgmaterials.cpp(2178): if (semValueOf < ProgramValue::FLOATS) { // INT VALUES
			//	ENABLED,
				USERDATA
			};

		private:

			SEMANTIC_TYPE m_Type;
			int m_ValueOf;
			nau::Enums::DataType m_ValueType;
			std::string m_Context, m_Name;
			std::string m_Param;
			float* m_Value;
			int *m_IntValue;
			int m_Cardinality;
			int m_Id;
			bool m_InSpecML; // true for values specified in the material library, false for other uniforms

		public:

			static bool Validate(std::string type,std::string context,std::string component);
			static std::string getSemanticTypeString(SEMANTIC_TYPE s);
			static const std::string semanticTypeString[];

			ProgramValue ();
			ProgramValue (std::string name, std::string type,std::string context,std::string valueof, int id, bool inSpecML = true);
			~ProgramValue();

			void clone(ProgramValue &pv);

			std::string getName();
			std::string getContext();
			void setContext(std::string s);

			int getCardinality ();
				
			bool isInSpecML();

			int getId();
			void setId(int id);
			nau::Enums::DataType getValueType ();
			SEMANTIC_TYPE getSemanticType();
			int getSemanticValueOf();
			void setSemanticType(SEMANTIC_TYPE s);
			void setSemanticValueOf(int s);
			void setValueType(nau::Enums::DataType s);
				
			void setValueOfUniform(int *values);
			void setValueOfUniform (float *values);
			void* getValues ();

		};
	};
};
#endif //PROGRAMVALUE_H
