#ifndef IUNIFORM_H
#define IUNIFORM_H

#include <string>
#include <nau/enums.h>


using namespace nau;

namespace nau {

	namespace render {
      
		class IUniform {
	 
		public:

			IUniform() {};
			void setName (std::string &name);
			std::string &getName (void);
			Enums::DataType getSimpleType();
			std::string getStringSimpleType();

		protected:
			std::string m_Name;
			Enums::DataType m_SimpleType;
		};
	};
};
#endif //IUNIFORM_H
