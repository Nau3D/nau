#ifndef TERRAIN_H
#define TERRAIN_H


#include "nau/geometry/primitive.h"

#include <string>

namespace nau
{
	namespace geometry
	{
		class Terrain : public Primitive
		{
		public:
			friend class nau::resource::ResourceManager;

			~Terrain(void);

			void setHeightMap(const std::string &name);

			void build();

		private:
			
			std::string m_HeightMap;

		protected:
			Terrain(void);


		};
	};
};
#endif
