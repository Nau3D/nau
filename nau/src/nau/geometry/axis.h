#ifndef AXIS_H
#define AXIS_H

#include "nau/geometry/mesh.h"
#include "nau/geometry/primitive.h"

namespace nau
{
	namespace geometry
	{
		class Axis : public Primitive
		{
		public:

			friend class nau::resource::ResourceManager;
			~Axis(void);

			std::string getClassName();

			void build();

		private:
			
			std::vector<float> m_Floats;

		protected:
			Axis();

		};
	};
};
#endif
