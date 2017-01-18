#ifndef PROGRAMVALUE_H
#define PROGRAMVALUE_H

#include <string>
#include "nau/enums.h"

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

namespace nau {

	namespace material {
      
		class ProgramValue {
	 
		public:
 
		private:
			std::string m_TypeString;
			int m_ValueOf;
			std::string m_ValueOfString;
			nau::Enums::DataType m_ValueType;
			std::string m_Context, m_Name;
			std::string m_Param;
			void *m_Values = NULL;
			int m_Cardinality = 0;
			int m_Id = -1;
			int m_Loc = -1;
			bool m_InSpecML; // true for values specified in the material library, false for other uniforms
			float m_fDummy;
		public:

			nau_API ProgramValue ();
			nau_API ProgramValue (std::string name, std::string type,std::string context,std::string valueof, int id, bool inSpecML = true);
			nau_API ~ProgramValue();

			void clone(ProgramValue &pv);

			nau_API const std::string &getName();
			nau_API const std::string &getType();
			nau_API const std::string &getContext();
			nau_API const std::string &getValueOf();
			nau_API void setContext(std::string s);

			nau_API int getCardinality ();
				
			nau_API bool isInSpecML();

			nau_API int getId();
			nau_API void setId(int id);
			nau_API nau::Enums::DataType getValueType ();
			nau_API int getSemanticValueOf();
			nau_API void setSemanticValueOf(int s);
			nau_API void setValueType(nau::Enums::DataType s);
				
			nau_API void setValueOfUniform(void *values);
			nau_API void* getValues ();

			nau_API void setLoc(int l);
			nau_API int getLoc();

		};
	};
};
#endif //PROGRAMVALUE_H
