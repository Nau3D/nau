#ifndef MATERIALID_H
#define MATERIALID_H

#include <string>
#include "nau/material/material.h"

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

		class MaterialID {

		private:
			std::string m_LibName, m_MatName;

		public:
			nau_API MaterialID (void);
			nau_API MaterialID (std::string libName, std::string matName);
			nau_API ~MaterialID (void);

			nau_API void setMaterialID (std::string libName, std::string matName);
			nau_API const std::string& getLibName (void);
			nau_API const std::string& getMaterialName (void);

			std::shared_ptr<Material> m_MatPtr;
			std::shared_ptr<Material> &getMaterialPtr();
		};
	};
};

#endif //MATERIALID_H
