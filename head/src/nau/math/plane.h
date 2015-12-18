#ifndef PLANE_H
#define PLANE_H

#include "nau/math/vec3.h"

namespace nau
{
	namespace math
	{
		class plane
		{
		private:
			vec3 normal;
			float d;

		public:
			plane(void);

			void setCoefficients (float a, float b, float c, float d);
			float distance (vec3 &v);

			const vec3& getNormal (void);
		public:
			~plane(void);
		};
	};
};
#endif //PLANE_H
