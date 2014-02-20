#include <nau/math/vec4.h>
#include <nau/math/utils.h>

namespace nau
{
	namespace math
	{

		// Tolerance factor to prevent division by 0
		static const double tol = FLT_EPSILON;

		// Private copy constructor
		vec4::vec4(const vec4 &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}


		// Private assignment operator
		const vec4& 
		vec4::operator =(const vec4 &v) 
		{
			if (this != &v) {
				this->x = v.x;
				this->y = v.y;
				this->z = v.z;
				this->w = v.w;
			}
			return *this;
		}

		// Make this vector a copy the vector <i>v</i>
		void 
		vec4::copy (const vec4 &v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			w = v.w;
		}
		   
		// Return a new vector with the same contents a this vector
		vec4 * 
		vec4::clone () const
		{
			return new vec4(*this);
		}

		// Initialize ou change the vector's components
		void 
		vec4::set(float x,float y, float z, float w) {
			this->x = x;
			this->y = y;
			this->z = z;
			this->w = w;
		}

		void 
		vec4::set(float *values) {
			this->x = values[0];
			this->y = values[1];
			this->z = values[2];
			this->w = values[3];
		}

		void 
		vec4::set(vec4* aVec ) {
			this->x = aVec->x;
			this->y = aVec->y;
			this->z = aVec->z;
			this->w = aVec->w;
		}	
		   
		void 
		vec4::set(const vec4& aVec ) {
			this->x = aVec.x;
			this->y = aVec.y;
			this->z = aVec.z;
			this->w = aVec.w;
		}	

		// Vector addition
		const vec4& 
		vec4::operator += (const vec4 &v) 
		{
			x += v.x;
			y += v.y;
			z += v.z;
			w += v.w;
		   
			return *this;
		}

		// Vector subtraction
		const vec4& 
		vec4::operator -= (const vec4 &v) 
		{
			x -= v.x;
			y -= v.y;
			z -= v.z;
			w -= v.w;

			return (*this);
		}
		   
		// Vector negation
		const vec4 & 
		vec4::operator - (void) 
		{
			x = -x;
			y = -y;
			z = -z;
			w = -w;
			
			return (*this);
		}

		// Vector scaling
		const vec4& 
		vec4::operator *=(float t) 
		{
			x *= t;
			y *= t;
			z *= t;
			w *= t;

			return *this;
		}

		// Scalar division
		const vec4& 
		vec4::operator /=(float t) 
		{
			float aux = 1/t;
		   
			x *= aux;
			y *= aux;
			z *= aux;
			w *= aux;
		   
			return *this;
		}

		// Equal
		bool 
		vec4::operator == (const vec4 &v) const
		{
			return equals(v);
		}
			
		//! Diferent
		bool 
		vec4::operator != (const vec4 &v) const
		{
			return !equals(v);
		}
		   
		// Length of vector
		float 
		vec4::length() const 
		{
			return sqrtf(x*x + y*y + z*z);// + w*w);
		}

		   
		// Length without the square root
		float 
		vec4::sqrLength() const 
		{
			return (x*x + y*y + z*z); // + w*w);
		}

		// distance between this vector and vector <i>v</i>
		float 
		vec4::distance(const vec4 &v) const 
		{
			vec4 aux;
		   
			aux.x = x - v.x;
			aux.y = y - v.y;
			aux.z = z - v.z;
			aux.w = w - v.w;
		   
			return sqrtf(aux.x*aux.x + aux.y*aux.y + aux.z*aux.z + aux.w*aux.w);
		}
			
		// Normalize this vector
		void 
		vec4::normalize() 
		{
			float m = length();
		   
			if(m <= tol) m = 1;
		   
			x /= m;
			y /= m;
			z /= m;
			w /= m;

			if (fabs(x) < tol) x = 0.0;
			if (fabs(y) < tol) y = 0.0;
			if (fabs(z) < tol) z = 0.0;
			if (fabs(w) < tol) w = 0.0;
		}
		   
		// Return the unit vector of this vector
		const vec4 
		vec4::unitVector() const 
		{   
			vec4 u(*this);
		   
			u.normalize();
		   
			return u;
		}

		// Dot product between this vector and vector <i>v</i>
		
		float 
		vec4::dot(const vec4 &v) const 
		{
			return (x * v.x + y * v.y + z * v.z + w * v.w);
		}
		

		// Cross product between this vector and vector <i>v</i>
		
		const vec4 
		vec4::cross(const vec4 &v) const
		{
		  vec4 result;
		  
		  result.x = (this->y * v.z) - (v.y * this->z);
		  result.y = (this->z * v.x) - (v.z * this->x);
		  result.z = (this->x * v.y) - (v.x * this->y);
		  result.w = 0.0f;

		  return result;
		}
		   
		// Angle between two vectors.
		/*
		float 
		vec4::angle(vec4 &v) const 
		{
			vec4 aux;
		   
			aux = this->unitVector();
			v.normalize();
		   
			return acosf(aux.dot(v));
		}
		*/
		// Interpolated vector between this vector and vector <i>v</i> at
		// position alpha (0.0 <= alpha <= 1.0)
	  
		const vec4 
		vec4::lerp (const vec4 &v, float alpha) const {
			vec4 result;

			float ix = x + ((v.x - x) * alpha);
			float iy = y + ((v.y - y) * alpha);
			float iz = z + ((v.z - z) * alpha);
			float iw = w + ((v.w - w) * alpha);

			result.set(ix, iy, iz, iw);

			return (result);
		}
		
		// Add another vector to this one
		void vec4::add (const vec4 &v)
		{
			x += v.x;
			y += v.y; 
			z += v.z;
			w += v.w;
		}

		// Scalar multiplication
		void 
		vec4::scale (float a)
		{				
			x *= a;
			y *= a;
			z *= a;
			w *= a;
		}
		   
		// Vector Equality
		bool 
		vec4::equals (const vec4 &v, float tolerance) const {
			return (FloatEqual(x, v.x, tolerance) && FloatEqual(y, v.y, tolerance) && \
				FloatEqual(z, v.z, tolerance) && FloatEqual(w, v.w, tolerance));
		}   
	};
};
