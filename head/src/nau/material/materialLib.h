#ifndef MATERIALLIB_H
#define MATERIALLIB_H

#include "nau/event/iListener.h"
#include "nau/material/material.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <string>


namespace nau
{
	namespace material
	{

		class MaterialLib: public nau::event_::IListener {

		private:
		   
			std::map<std::string, std::shared_ptr<Material>> m_MaterialLib;
			std::string m_LibName;

			std::shared_ptr<Material> p_Default;

		public:
			MaterialLib (const std::string &libName);
			~MaterialLib();

			void eventReceived(const std::string &sender, const std::string &eventType, 
				const std::shared_ptr<nau::event_::IEventData> &evt);

			void clear();
			std::string &getName();
			
			bool hasMaterial (const std::string &materialName);
			void addMaterial (std::shared_ptr<Material> &aMaterial);
			std::shared_ptr<Material> &getMaterial(const std::string &);
			
			int getMaterialCount();
			void getMaterialNames(std::vector<std::string>* ret);
			void getMaterialNames(const std::string &aName, std::vector<std::string>* ret);
		};
	};
};
#endif //MATERIALLIB_H
