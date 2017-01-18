#ifndef IUNIFORM_H
#define IUNIFORM_H

#include <string>
#include "nau/enums.h"


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


namespace nau {

	namespace material {
      
		class IUniform {
	 
		public:

			nau_API IUniform() {};
			nau_API void setName (std::string &name);
			nau_API std::string &getName (void);
			nau_API Enums::DataType getSimpleType();
			nau_API std::string getStringSimpleType();

		protected:
			std::string m_Name;
			Enums::DataType m_SimpleType;
		};
	};
};
#endif //IUNIFORM_H
