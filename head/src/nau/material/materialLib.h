#ifndef MATERIALLIB_H
#define MATERIALLIB_H

#include <string>
#include <map>
#include <fstream>
#include <iostream>

#include "nau/event/ilistener.h"
#include "nau/material/material.h"

//include "filenames.h"

namespace nau
{
	namespace material
	{

		class MaterialLib: public nau::event_::IListener {

		private:
		   
			std::map<std::string, nau::material::Material*> m_MaterialLib;
			std::string m_LibName;

			Material p_Default;

			//std::string m_Filename;

		public:
			MaterialLib (std::string libName);
			~MaterialLib();

			void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);

			void clear();
			std::string &getName();
			
			bool hasMaterial (std::string materialName);
			void addMaterial (nau::material::Material* aMaterial); /***MARK***/ //To be removed, probably
			nau::material::Material* getMaterial(std::string s);
			
			void getMaterialNames(std::vector<std::string>* ret);
			void getMaterialNames(const std::string &aName, std::vector<std::string>* ret);

			//void load(std::string &filename);
			//void save(std::string path);
			//void save(std::ofstream &outf, std::string path);

			//void add(Material *mat);
		};
	};
};
#endif //MATERIALLIB_H
