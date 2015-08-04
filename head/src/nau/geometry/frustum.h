#ifndef FRUSTUM_H
#define FRUSTUM_H

#include "nau/math/vec3.h"
#include "nau/math/plane.h"
#include "nau/scene/camera.h"
#include "nau/geometry/iBoundingVolume.h"



namespace nau
{
	namespace geometry
	{
		class Frustum
		{
		private:
			nau::math::plane m_Planes[6];
			nau::math::vec3 m_Points[8];

		private:
			enum {
				TOP = 0,
				BOTTOM,
				LEFT,
				RIGHT,
				NEARP,
				FARP
			}FrustumPlanes;

		public:
			enum {
				OUTSIDE, 
				INTERSECT, 
				INSIDE
			}FrustumTest;

			/// Constructs the frustum from a PVM matrix
			void setFromMatrix (const float *m);
			/// determines if the volume is inside the frustum using all 6 planes
			/// if conservative is set to true, the near and far planes are not tested
			int isVolumeInside (const nau::geometry::IBoundingVolume *aBoundingVolume, bool conservative=false);


		public:
			~Frustum(void);

		public:
			Frustum (void);
		};
	};
};

#endif
