#ifndef MATERIALLIBMANAGER_H
#define MATERIALLIBMANAGER_H

#include <string>
#include <map>

#include "nau/material/materialId.h"
#include "nau/material/materialLib.h"

using namespace nau::material;

#define DEFAULTMATERIALLIBNAME "__nau_material_lib"

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

		class MaterialLibManager {

		private:
			std::map<std::string, nau::material::MaterialLib*> m_LibManager;
			MaterialLib *m_DefaultLib;
			MaterialLibManager(const MaterialLibManager&);


		public:
			nau_API MaterialLibManager();
			nau_API ~MaterialLibManager();

			nau_API void clear();

			nau_API bool hasLibrary (const std::string &lib);
			nau_API nau::material::MaterialLib* getLib(const std::string &libName);

			nau_API bool hasMaterial (const std::string &aLibrary, const std::string &name);
			nau_API bool hasMaterial(const std::string &amat);
			nau_API void addMaterial (const std::string &aLibrary, std::shared_ptr<Material> &aMaterial);
			nau_API std::shared_ptr<Material> &getMaterialFromDefaultLib(const std::string &name);
			nau_API std::shared_ptr<Material> &getMaterial(nau::material::MaterialID &materialID);
			nau_API std::shared_ptr<Material> &getMaterial (const std::string &lib, const std::string &material);
			nau_API std::shared_ptr<Material> &getMaterial(const std::string &fullMatName);

			nau_API std::shared_ptr<Material> createMaterial(const std::string &lib, const std::string &material);
			nau_API std::shared_ptr<Material> createMaterial(const std::string &material);
			nau_API std::shared_ptr<Material> cloneMaterial(std::shared_ptr<Material> &);

			nau_API void getLibNames (std::vector<std::string>* );
			nau_API void getNonEmptyLibNames(std::vector<std::string>*);
			nau_API void getMaterialNames (const std::string &lib, std::vector<std::string> *ret);
			
			nau_API unsigned int getNumLibs();
		};
	};
};

#endif
