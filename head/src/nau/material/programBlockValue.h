#ifndef PROGRAMBLOCKVALUE_H
#define PROGRAMBLOCKVALUE_H

#include <string>
#include "nau/enums.h"

namespace nau {

	namespace material {
      
		class ProgramBlockValue {
	 
		public:
 
		private:
			std::string m_TypeString;
			std::string m_ValueOfString;
			int m_ValueOf;
			int m_Id;
			nau::Enums::DataType m_ValueType;
			void *m_Values;
			std::string m_Context, m_Name;
			std::string m_Block;
			std::string m_Param;
			int m_Cardinality = 0;
			bool m_InSpecML; // true for values specified in the material library, false for other uniforms
			float m_fDummy;
		public:

			ProgramBlockValue ();
			ProgramBlockValue (std::string name, std::string block, std::string type,std::string context,std::string valueof, int id, bool inSpecML = true);
			~ProgramBlockValue();

			void clone(ProgramBlockValue &pv);

			const std::string &getName();
			const std::string &getType();
			const std::string &getContext();
			const std::string &getValueOf();
			void setContext(std::string s);

			int getCardinality ();
			bool isInSpecML();

			int getId();
			void setId(int id);
			nau::Enums::DataType getValueType ();
//			SEMANTIC_TYPE getSemanticType();
			int getSemanticValueOf();
			//void setSemanticType(SEMANTIC_TYPE s);
			void setSemanticValueOf(int s);
			void setValueType(nau::Enums::DataType s);
				
			void setValueOfUniform(void *values);
			void* getValues ();
		};
	};
};
#endif 
