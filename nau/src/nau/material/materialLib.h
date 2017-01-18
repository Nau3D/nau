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
			nau_API MaterialLib (const std::string &libName);
			nau_API~MaterialLib();

			void eventReceived(const std::string &sender, const std::string &eventType, 
				const std::shared_ptr<nau::event_::IEventData> &evt);

			nau_API void clear();
			nau_API std::string &getName();
			
			nau_API bool hasMaterial (const std::string &materialName);
			nau_API void addMaterial (std::shared_ptr<Material> &aMaterial);
			nau_API std::shared_ptr<Material> &getMaterial(const std::string &);
			
			nau_API unsigned int getMaterialCount();
			nau_API void getMaterialNames(std::vector<std::string>* ret);
			nau_API void getMaterialNames(const std::string &aName, std::vector<std::string>* ret);
		};
	};
};
#endif //MATERIALLIB_H
