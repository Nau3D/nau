#ifndef SPHERICAL_H
#define SPHERICAL_H

#include "nau/math/vec3.h"
#include "nau/math/vec2.h"

namespace nau
{
	namespace math
	{

		//! \brief This class converts between spherical and cartesian coordinates
		//!
		//!	
		class Spherical {
		 public:
		  		  
			static vec3 toCartesian(float alpha, float beta);
			static vec2 toSpherical(float x, float y, float z);
			static vec3 getRightVector(float alpha, float beta);
			static vec3 getNaturalUpVector(float alpha, float beta);
			static float capBeta(float beta);

		protected:
			Spherical () {};
		   


		};
	};
};
#endif // spherical_H
