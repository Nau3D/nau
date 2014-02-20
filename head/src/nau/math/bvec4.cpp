#include <nau/math/bvec4.h>

namespace nau
{
	namespace math
	{

		// Private copy constructor
		bvec4::bvec4(const bvec4 &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}


		// Private assignment operator
		const bvec4& 
		bvec4::operator =(const bvec4 &v) 
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
		bvec4::copy (const bvec4 &v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			w = v.w;
		}
		   
		// Return a new vector with the same contents a this vector
		bvec4 * 
		bvec4::clone () const
		{
			return new bvec4(*this);
		}

		// Initialize ou change the vector's components
		void 
		bvec4::set(bool x,bool y, bool z, bool w) {
			this->x = x;
			this->y = y;
			this->z = z;
			this->w = w;
		}

		void 
		bvec4::set(bool *values) {
			this->x = values[0];
			this->y = values[1];
			this->z = values[2];
			this->w = values[3];
		}

		void 
		bvec4::set(bvec4* aVec ) {
			this->x = aVec->x;
			this->y = aVec->y;
			this->z = aVec->z;
			this->w = aVec->w;
		}	
		   
		void 
		bvec4::set(const bvec4& aVec ) {
			this->x = aVec.x;
			this->y = aVec.y;
			this->z = aVec.z;
			this->w = aVec.w;
		}	


		// Equal
		bool 
		bvec4::operator == (const bvec4 &v) const
		{
			return (this->x == v.x && this->y == v.y && this->z == v.z && this->w == v.w);
		}
			
		//! Diferent
		bool 
		bvec4::operator != (const bvec4 &v) const
		{
			return (this->x != v.x || this->y != v.y || this->z != v.z || this->w != v.w);
		}
		   
	};
};
