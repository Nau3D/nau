#include <nau/math/vec2.h>
#include <nau/math/utils.h>

namespace nau
{
	namespace math
	{

		// Tolerance factor to prevent division by 0
		static const double tol = FLT_EPSILON;

	  // Static copies of the base unit vectors
	  const vec2 vec2::UNIT_X (1.0f, 0.0f);
	  const vec2 vec2::UNIT_Y (0.0f, 1.0f);
	  const vec2 vec2::NEGATIVE_UNIT_X (-1.0f, 0.0f);
	  const vec2 vec2::NEGATIVE_UNIT_Y (0.0f, -1.0);

		// Private copy constructor
		vec2::vec2(const vec2 &v) : x(v.x), y(v.y) {}


		// Private assignment operator
		const vec2& 
		vec2::operator =(const vec2 &v) 
		{
			if (this != &v) {
				this->x = v.x;
				this->y = v.y;
			}
			return *this;
		}

		// Make this vector a copy the vector <i>v</i>
		void 
		vec2::copy (const vec2 &v)
		{
			x = v.x;
			y = v.y;
		}
		   
		// Return a new vector with the same contents a this vector
		vec2 * 
		vec2::clone () const
		{
			return new vec2(*this);
		}

		// Initialize ou change the vector's components
		void 
		vec2::set(float x,float y) {
			this->x = x;
			this->y = y;
		}	
		   

		// Vector addition
		const vec2& 
		vec2::operator += (const vec2 &v) 
		{
			x += v.x;
			y += v.y;
		   
			return *this;
		}

		// Vector subtraction
		const vec2& 
		vec2::operator -= (const vec2 &v) 
		{
			x -= v.x;
			y -= v.y;

			return (*this);
		}
		   
		// Vector negation
		const vec2 & 
		vec2::operator - (void) 
		{
			x = -x;
			y = -y;
			
			return (*this);
		}

		// Vector scaling
		const vec2& 
		vec2::operator *=(float t) 
		{
			x *= t;
			y *= t;

			return *this;
		}

		// Scalar division
		const vec2& 
		vec2::operator /=(float t) 
		{
			float aux = 1/t;
		   
			x *= aux;
			y *= aux;
		   
			return *this;
		}

		// Equal
		bool 
		vec2::operator == (const vec2 &v) const
		{
			return equals(v);
		}
			
		//! Diferent
		bool 
		vec2::operator != (const vec2 &v) const
		{
			return !equals(v);
		}
		   
		// Length of vector
		float 
		vec2::length() const 
		{
			return sqrtf(x*x + y*y);
		}

		   
		// Length without the square root
		float 
		vec2::sqrLength() const 
		{
			return (x*x + y*y);
		}

		// distance between this vector and vector <i>v</i>
		float 
		vec2::distance(const vec2 &v) const 
		{
			vec2 aux;
		   
			aux.x = x - v.x;
			aux.y = y - v.y;
		   
			return sqrtf(aux.x*aux.x + aux.y*aux.y);
		}
			
		// Normalize this vector
		void 
		vec2::normalize() 
		{
			float m = length();
		   
			if(m <= tol) m = 1;
		   
			x /= m;
			y /= m;

		}
		   
		// Return the unit vector of this vector
		const vec2 
		vec2::unitVector() const 
		{   
			vec2 u(*this);
		   
			u.normalize();
		   
			return u;
		}

		// Dot product between this vector and vector <i>v</i>
		float 
		vec2::dot(const vec2 &v) const 
		{
			return (x * v.x + y * v.y);
		}
		   
		   
		// Angle between two vectors.
		float 
		vec2::angle(vec2 &v) const 
		{
			vec2 aux;
		   
			aux = this->unitVector();
			v.normalize();
		   
			return acosf(aux.dot(v));
		}

		// Interpolated vector between this vector and vector <i>v</i> at
		// position alpha (0.0 <= alpha <= 1.0)
		const vec2 
		vec2::lerp (const vec2 &v, float alpha) const {
			vec2 result;

			float ix = x + ((v.x - x) * alpha);
			float iy = y + ((v.y - y) * alpha);

			result.set(ix, iy);

			return (result);
		}

		// Add another vector to this one
		void vec2::add (const vec2 &v)
		{
			x += v.x;
			y += v.y; 
		}

		// Scalar multiplication
		void 
		vec2::scale (float a)
		{				
			x *= a;
			y *= a;
		}
		   
		// Vector Equality
		bool 
		vec2::equals (const vec2 &v, float tolerance) const {
			return (FloatEqual(x, v.x, tolerance) && FloatEqual(y, v.y, tolerance));
		}   
	};
};
