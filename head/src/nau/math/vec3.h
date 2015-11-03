#ifndef NAU_VECTOR3_H
#define NAU_VECTOR3_H

#include "nau/math/data.h"
#include "nau/math/utils.h"

#include <string>

namespace nau
{
	namespace math
	{
		template <typename T>
		class vector3: public Data {

		public:

			T x, y, z;

			vector3() : x(0), y(0), z(0) {};
			vector3(T x, T y, T z) : x(x), y(y), z(z) {};
			vector3(T v) : x(v), y(v), z(v) {};
			vector3(const vector3 &v) : x(v.x), y(v.y), z(v.z) {};
			~vector3() {};

			const vector3&
				vector3::operator =(const vector3 &v) {

				if (this != &v) {
					x = v.x;
					y = v.y;
					z = v.z;
				}
				return *this;
			};

			void
				copy(const vector3 &v) {

				x = v.x;
				y = v.y;
				z = v.z;
			};

			vector3 *
				clone() const {

				return new vector(*this);
			};

			void
				set(T xx, T yy, T zz) {
				x = xx;
				y = yy;
				z = zz;
			};

			void
				set(T *values) {
				x = values[0];
				y = values[1];
				z = values[2];
			};

			void
				set(vector3* aVec) {
				x = aVec->x;
				y = aVec->y;
				z = aVec->z;
			};

			void
				set(const vector3& aVec) {
				x = aVec.x;
				y = aVec.y;
				z = aVec.z;
			};

			const vector3&
				operator += (const vector3 &v)	{

				x += v.x;
				y += v.y;
				z += v.z;
				return *this;
			};

			const vector3&
				operator -= (const vector3 &v) {

				x -= v.x;
				y -= v.y;
				z -= v.z;
				return (*this);
			};

			const vector3 &
				operator - (void) {

				x = -x;
				y = -y;
				z = -z;
				return (*this);
			};

			const vector3&
				operator *=(T t) {

				x *= t;
				y *= t;
				z *= t;
				return *this;
			};

			const vector3&
			operator /=(T t) {

				x /= t;
				y /= t;
				z /= t;
				return *this;
			};

			bool
			operator == (const vector3 &v) const {

				return equals(v);
			};

			bool
			operator != (const vector3 &v) const {

				return !equals(v);
			};

			bool 
				operator > (const vector3 &v) const {

				if (x > v.x && y > v.y && z > v.z)
					return true;
				else
					return false;
			}

			bool
				operator < (const vector3 &v) const {

				if (x < v.x && y < v.y && z < v.z)
					return true;
				else
					return false;
			}


			void
				add(const vector3 &v) {
				x += v.x;
				y += v.y;
				z += v.z;
			};

			void
				scale(T a) {

				x *= a;
				y *= a;
				z *= a;
			};

			float
				length() const {

				return sqrtf(x*x + y*y + z*z);
			};

			void
				normalize() {

				float m = length();
				if (m <= FLT_EPSILON) {
					m = 1;
				}
				x /= m;
				y /= m;
				z /= m;
			};

			float
				dot(const vector3 &v) const {

				return (x*v.x + y*v.y + z*v.z);
			};

			const vector3
				cross(const vector3 &v) const {

				vector3 result;

				result.x = (this->y * v.z) - (v.y * this->z);
				result.y = (this->z * v.x) - (v.z * this->x);
				result.z = (this->x * v.y) - (v.x * this->y);

				return result;
			};

			bool
			between(const vector3 &v1, const vector3 &v2) {

				if (x < v1.x || x > v2.x)
					return false;
				if (y < v1.y || y > v2.y)
					return false;
				if (z < v1.z || z > v2.z)
					return false;

				return true;
			};


			bool
				equals(const vector3 &v, float tolerance = -1.0f) const {
				return (FloatEqual(x, v.x, tolerance) && FloatEqual(y, v.y, tolerance) && \
					FloatEqual(z, v.z, tolerance));
			};

			std::string 
				toString() {

				return  "(" + std::to_string(x) + ", " + std::to_string(y)+ ", " + std::to_string(z) + ")";
			};
		};

		typedef vector3<float> vec3;
		typedef vector3<int> ivec3;
		typedef vector3 < unsigned int >  uivec3;
		typedef vector3<bool> bvec3;
		typedef vector3<double> dvec3;
	};
};


#endif 
