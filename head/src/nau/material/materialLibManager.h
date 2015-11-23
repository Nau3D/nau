#ifndef MATERIALLIBMANAGER_H
#define MATERIALLIBMANAGER_H

#include <string>
#include <map>

#include "nau/material/materialId.h"
#include "nau/material/materialLib.h"

using namespace nau::material;

// FIXME: Don't know where this should be defined. Added it here for now
#define DEFAULTMATERIALLIBNAME "nau_material_lib"

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
			MaterialLibManager();
			~MaterialLibManager();

			void clear();

			bool hasLibrary (const std::string &lib);
			nau::material::MaterialLib* getLib(const std::string &libName);

			bool hasMaterial (const std::string &aLibrary, const std::string &name);
			void addMaterial (const std::string &aLibrary, std::shared_ptr<Material> &aMaterial);
			std::shared_ptr<Material> &getMaterialFromDefaultLib(const std::string &name);
			std::shared_ptr<Material> &getMaterial(nau::material::MaterialID &materialID);
			std::shared_ptr<Material> &getMaterial (const std::string &lib, const std::string &material);

			std::shared_ptr<Material> createMaterial(const std::string &lib, const std::string &material);
			std::shared_ptr<Material> createMaterial(const std::string &material);
			std::shared_ptr<Material> cloneMaterial(std::shared_ptr<Material> &);

			void getLibNames (std::vector<std::string>* );
			void getNonEmptyLibNames(std::vector<std::string>*);
			void getMaterialNames (const std::string &lib, std::vector<std::string> *ret);
			
			unsigned int getNumLibs();
		};
	};
};

#endif
