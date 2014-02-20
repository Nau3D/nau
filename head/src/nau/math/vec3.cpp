#include <nau/math/vec3.h>
#include <nau/math/utils.h>

namespace nau
{
	namespace math
	{

		// Tolerance factor to prevent division by 0
		static const double tol = FLT_EPSILON;

	  // Static copies of the base unit vectors
	  const vec3 vec3::UNIT_X (1.0f, 0.0f, 0.0f);
	  const vec3 vec3::UNIT_Y (0.0f, 1.0f, 0.0f);
	  const vec3 vec3::UNIT_Z (0.0f, 0.0f, 1.0f);
	  const vec3 vec3::NEGATIVE_UNIT_X (-1.0f, 0.0f, 0.0f);
	  const vec3 vec3::NEGATIVE_UNIT_Y (0.0f, -1.0f, 0.0f);
	  const vec3 vec3::NEGATIVE_UNIT_Z (0.0f, 0.0f, -1.0f);

		// Private copy constructor
		vec3::vec3(const vec3 &v) : x(v.x), y(v.y), z(v.z) {}


		// Private assignment operator
		const vec3& 
		vec3::operator =(const vec3 &v) 
		{
			if (this != &v) {
				this->x = v.x;
				this->y = v.y;
				this->z = v.z;
			}
			return *this;
		}

		// Make this vector a copy the vector <i>v</i>
		void 
		vec3::copy (const vec3 &v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
		}
		   
		// Return a new vector with the same contents a this vector
		vec3 * 
		vec3::clone () const
		{
			return new vec3(*this);
		}

		// Initialize ou change the vector's components
		void 
		vec3::set(float x,float y, float z) {
			this->x = x;
			this->y = y;
			this->z = z;
		}	
		   

		// Vector addition
		const vec3& 
		vec3::operator += (const vec3 &v) 
		{
			x += v.x;
			y += v.y;
			z += v.z;
		   
			return *this;
		}

		// Vector subtraction
		const vec3& 
		vec3::operator -= (const vec3 &v) 
		{
			x -= v.x;
			y -= v.y;
			z -= v.z;

			return (*this);
		}
		   
		// Vector negation
		const vec3 & 
		vec3::operator - (void) 
		{
			x = -x;
			y = -y;
			z = -z;
			
			return (*this);
		}

		// Vector scaling
		const vec3& 
		vec3::operator *=(float t) 
		{
			x *= t;
			y *= t;
			z *= t;

			return *this;
		}

		// Scalar division
		const vec3& 
		vec3::operator /=(float t) 
		{
			float aux = 1/t;
		   
			x *= aux;
			y *= aux;
			z *= aux;
		   
			return *this;
		}

		// Equal
		bool 
		vec3::operator == (const vec3 &v) const
		{
			return equals(v);
		}
			
		//! Diferent
		bool 
		vec3::operator != (const vec3 &v) const
		{
			return !equals(v);
		}
		   
		// Length of vector
		float 
		vec3::length() const 
		{
			return sqrtf(x*x + y*y + z*z);
		}

		   
		// Length without the square root
		float 
		vec3::sqrLength() const 
		{
			return (x*x + y*y + z*z);
		}

		// distance between this vector and vector <i>v</i>
		float 
		vec3::distance(const vec3 &v) const 
		{
			vec3 aux;
		   
			aux.x = x - v.x;
			aux.y = y - v.y;
			aux.z = z - v.z;
		   
			return sqrtf(aux.x*aux.x + aux.y*aux.y + aux.z*aux.z);
		}
			
		// Normalize this vector
		void 
		vec3::normalize() 
		{
			float m = length();
		   
			if(m <= tol) m = 1;
		   
			x /= m;
			y /= m;
			z /= m;

		//	if (fabs(x) < tol) x = 0.0;
		//	if (fabs(y) < tol) y = 0.0;
		//	if (fabs(z) < tol) z = 0.0;
		}
		   
		// Return the unit vector of this vector
		const vec3 
		vec3::unitVector() const 
		{   
			vec3 u(*this);
		   
			u.normalize();
		   
			return u;
		}

		// Dot product between this vector and vector <i>v</i>
		float 
		vec3::dot(const vec3 &v) const 
		{
			return (x * v.x + y * v.y + z * v.z);
		}
		   
		// Cross product between this vector and vector <i>v</i>
		const vec3 
		vec3::cross(const vec3 &v) const
		{
		  vec3 result;
		  
		  result.x = (this->y * v.z) - (v.y * this->z);
		  result.y = (this->z * v.x) - (v.z * this->x);
		  result.z = (this->x * v.y) - (v.x * this->y);

		  return result;
		}
		   
		// Angle between two vectors.
		float 
		vec3::angle(vec3 &v) const 
		{
			vec3 aux;
		   
			aux = this->unitVector();
			v.normalize();
		   
			return acosf(aux.dot(v));
		}

		// Interpolated vector between this vector and vector <i>v</i> at
		// position alpha (0.0 <= alpha <= 1.0)
		const vec3 
		vec3::lerp (const vec3 &v, float alpha) const {
			vec3 result;

			float ix = x + ((v.x - x) * alpha);
			float iy = y + ((v.y - y) * alpha);
			float iz = z + ((v.z - z) * alpha);

			result.set(ix, iy, iz);

			return (result);
		}

		// Add another vector to this one
		void vec3::add (const vec3 &v)
		{
			x += v.x;
			y += v.y; 
			z += v.z;
		}

		// Scalar multiplication
		void 
		vec3::scale (float a)
		{				
			x *= a;
			y *= a;
			z *= a;
		}
		   
		// Vector Equality
		bool 
		vec3::equals (const vec3 &v, float tolerance) const {
			return (FloatEqual(x, v.x, tolerance) && FloatEqual(y, v.y, tolerance) && \
				FloatEqual(z, v.z, tolerance));
		}   
	};
};
