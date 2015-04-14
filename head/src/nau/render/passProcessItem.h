#ifndef PASS_PROCESS_ITEM_H
#define PASS_PROCESS_ITEM_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"

namespace nau {
	namespace render {
	
		class PassProcessItem : public AttributeValues {

		public:

			virtual void process() = 0;
			virtual void setItemName(std::string &name) {

				m_Name = name;
			}

		protected:

			std::string m_Name;
		};
	};
};


#endif

