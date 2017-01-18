#ifndef _NAU_PHYSICS_MATERIAL
#definr _NAU_PHYSICS_MATERIAL

namespace nau 
{
	namespace physics 
	{
		class PhysicsMaterial: public AttributeValues 
		{
			// to allow pluggins to add properties
			void addProperty(std::string &property, Enums::Data dataType, Data *value);
		}
	}
}	
