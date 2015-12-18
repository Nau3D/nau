#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "nau/geometry/mesh.h"

namespace nau
{
	namespace geometry
	{
		class Primitive : public Mesh
		{
		public:
			virtual void build() = 0;
		};
	};
};
#endif
