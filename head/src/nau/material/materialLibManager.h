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


		public:
			MaterialLibManager();
			~MaterialLibManager();

			void clear();

			bool hasLibrary (std::string lib);
			nau::material::MaterialLib* getLib(std::string libName);

			bool hasMaterial (std::string aLibrary, std::string name);
			void addMaterial (std::string aLibrary, nau::material::Material* aMaterial);
			Material* getDefaultMaterial (std::string name);
			Material* getMaterial (nau::material::MaterialID &materialID);
			Material* getMaterial (std::string lib, std::string material);

			Material* createMaterial(std::string lib, std::string material);
			Material* createMaterial(std::string material);


			void getLibNames (std::vector<std::string>* );
			void getMaterialNames (const std::string &lib, std::vector<std::string> *ret);
			
			unsigned int getNumLibs();

		private:
			MaterialLibManager(const MaterialLibManager&);
			MaterialLibManager& operator= (const MaterialLibManager&);

			// adds some materials 
			//void addOwnMaterials();
		};
	};
};

#endif
