#ifndef SPHERE_H
#define SPHERE_H


#include "nau/geometry/primitive.h"

namespace nau
{
	namespace geometry
	{
		class Sphere : public Primitive
		{
		public:

			Sphere(void);			
			~Sphere(void);

			static AttribSet Attribs;
			
			FLOAT_PROP(SLICES, 0);
			FLOAT_PROP(STACKS, 1);

			void build();

		protected:

			std::vector<float> m_Floats;
			static bool Init();
			static bool Inited;

		};
	};
};
#endif
