#ifndef VECTOR4_H
#define VECTOR4_H

#include <nau/math/utils.h>

namespace nau
{
	namespace math
	{
		template <typename T>
		class vector4 {

		public:

			T x, y, z, w;

			vector4() : x(0), y(0), z(0), w(0) {};
			vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {};
			vector4(const vector4 &v) : x(v.x), y(v.y), z(v.z), w(v.w) {};
			~vector4() {};

			const vector4&
			vector4::operator =(const vector4 &v) {

				if (this != &v) {
					x = v.x;
					y = v.y;
					z = v.z;
					w = v.w;
				}
				return *this;
			};

			void
			copy(const vector4 &v) {

				x = v.x;
				y = v.y;
				z = v.z;
				w = v.w;
			};

			vector4 *
			clone() const {

				return new vector(*this);
			};

			void
			set(T xx, T yy, T zz, T ww = 1) {
				x = xx;
				y = yy;
				z = zz;
				w = ww;
			};

			void
			set(T *values) {
				x = values[0];
				y = values[1];
				z = values[2];
				w = values[3];
			};

			void
			set(vector4* aVec) {
				x = aVec->x;
				y = aVec->y;
				z = aVec->z;
				w = aVec->w;
			};

			void
			set(const vector4& aVec) {
				x = aVec.x;
				y = aVec.y;
				z = aVec.z;
				w = aVec.w;
			};

			const vector4&
			operator += (const vector4 &v)	{

				x += v.x;
				y += v.y;
				z += v.z;
				w += v.w;
				return *this;
			};

			const vector4&
			operator -= (const vector4 &v) {

				x -= v.x;
				y -= v.y;
				z -= v.z;
				w -= v.w;
				return (*this);
			};

			const vector4 &
			operator - (void) {

				x = -x;
				y = -y;
				z = -z;
				w = -w;
				return (*this);
			};

			const vector4&
			operator *=(T t) {

				x *= t;
				y *= t;
				z *= t;
				w *= t;
				return *this;
			};

			const vector4&
			operator /=(T t) {

				x /= t;
				y /= t;
				z /= t;
				w /= t;
				return *this;
			};

			bool
			operator == (const vector4 &v) const {

				return equals(v);
			};

			bool
				operator > (const vector4 &v) const {

				if (x > v.x && y > v.y && z > v.z && w > v.w)
					return true;
				else
					return false;
			}

			bool
				operator < (const vector4 &v) const {

				if (x < v.x && y < v.y && z < v.z && w < v.w)
					return true;
				else
					return false;
			}

			bool
			operator != (const vector4 &v) const {

				return !equals(v);
			};

			const vector4
			lerp(const vector4 &v, T alpha) const {

				vector result;

				T ix = x + ((v.x - x) * alpha);
				T iy = y + ((v.y - y) * alpha);
				T iz = z + ((v.z - z) * alpha);
				T iw = w + ((v.w - w) * alpha);

				result.set(ix, iy, iz, iw);

				return (result);
			};

			void
			add(const vector4 &v) {
				x += v.x;
				y += v.y;
				z += v.z;
				w += v.w;
			};

			void
			scale(T a) {

				x *= a;
				y *= a;
				z *= a;
				w *= a;
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
			dot(const vector4 &v) const {

				return (x*v.x + y*v.y + z*v.z);
			};

			const vector4
			cross(const vector4 &v) const {

				vector4 result;

				result.x = (this->y * v.z) - (v.y * this->z);
				result.y = (this->z * v.x) - (v.z * this->x);
				result.z = (this->x * v.y) - (v.x * this->y);
				result.w = 0.0f;

				return result;
			};

			bool 
			between(const vector4 &v1, const vector4 &v2) {

				if (x < v1.x || x > v2.x)
					return false;
				if (y < v1.y || y > v2.y)
					return false;
				if (z < v1.z || z > v2.z)
					return false;
				if (w < v1.w || w > v2.w)
					return false;

				return true;
			};


			bool
				equals(const vector4 &v, float tolerance = -1.0f) const {
				return (FloatEqual(x, v.x, tolerance) && FloatEqual(y, v.y, tolerance) && \
					FloatEqual(z, v.z, tolerance) && FloatEqual(w, v.w, tolerance));
			};
		};

		typedef vector4<float> vec4;
		typedef vector4<int> ivec4;
		typedef vector4 < unsigned int >  uivec4;
		typedef vector4<bool> bvec4;
		typedef vector4<double> dvec4;
	};
};


#endif 
