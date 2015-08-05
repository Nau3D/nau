#ifndef MATERIALLIB_H
#define MATERIALLIB_H

#include <string>
#include <map>
#include <fstream>
#include <iostream>

#include "nau/material/material.h"

//include "filenames.h"

namespace nau
{
	namespace material
	{

		class MaterialLib {

		private:
		   
			std::map<std::string, nau::material::Material*> m_MaterialLib;
			std::string m_LibName;

			Material p_Default;

			//std::string m_Filename;

		public:
			MaterialLib (std::string libName);
			~MaterialLib();

			void clear();
			std::string getName();
			
			bool hasMaterial (std::string materialName);
			void addMaterial (nau::material::Material* aMaterial); /***MARK***/ //To be removed, probably
			nau::material::Material* getMaterial(std::string s);
			
			std::vector<std::string>* getMaterialNames();
			std::vector<std::string>* getMaterialNames(std::string aName);

			//void load(std::string &filename);
			//void save(std::string path);
			//void save(std::ofstream &outf, std::string path);

			//void add(Material *mat);
		};
	};
};
#endif //MATERIALLIB_H
