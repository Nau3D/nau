#include <nau/math/vec2.h>
#include <nau/math/utils.h>

using namespace nau::math;

// Tolerance factor to prevent division by 0
static const double tol = FLT_EPSILON;


vec2::vec2() : x(0.0f), y(0.0f) {

}


vec2::vec2(float xx, float yy) : x(xx), y(yy) {

}


vec2::vec2(const vec2 &v) : x(v.x), y(v.y) {

}


const vec2& 
vec2::operator =(const vec2 &v) 
{
	if (this != &v) {
		this->x = v.x;
		this->y = v.y;
	}
	return *this;
}


void 
vec2::copy (const vec2 &v)
{
	x = v.x;
	y = v.y;
}
		   

vec2 * 
vec2::clone () const
{
	return new vec2(*this);
}


void 
vec2::set(float x,float y) {
	this->x = x;
	this->y = y;
}	
		   


const vec2& 
vec2::operator += (const vec2 &v) 
{
	x += v.x;
	y += v.y;
		   
	return *this;
}


const vec2& 
vec2::operator -= (const vec2 &v) 
{
	x -= v.x;
	y -= v.y;

	return (*this);
}
		   

const vec2 & 
vec2::operator - (void) 
{
	x = -x;
	y = -y;
			
	return (*this);
}


const vec2& 
vec2::operator *=(float t) 
{
	x *= t;
	y *= t;

	return *this;
}


const vec2& 
vec2::operator /=(float t) 
{
	float aux = 1/t;
		   
	x *= aux;
	y *= aux;
		   
	return *this;
}


bool 
vec2::operator == (const vec2 &v) const
{
	return equals(v);
}
			

bool 
vec2::operator != (const vec2 &v) const
{
	return !equals(v);
}


bool
vec2::operator > (const vec2 &v) {

	if (x > v.x && y > v.y)
		return true;
	else
		return false;
}

		   
bool
vec2::operator < (const vec2 &v) {

	if (x < v.x && y < v.y)
		return true;
	else
		return false;
}


float 
vec2::length() const 
{
	return sqrtf(x*x + y*y);
}

		   

float 
vec2::sqrLength() const 
{
	return (x*x + y*y);
}


float 
vec2::distance(const vec2 &v) const 
{
	vec2 aux;
		   
	aux.x = x - v.x;
	aux.y = y - v.y;
		   
	return sqrtf(aux.x*aux.x + aux.y*aux.y);
}
			

void 
vec2::normalize() 
{
	float m = length();
		   
	if(m <= tol) m = 1;
		   
	x /= m;
	y /= m;

}
		   

const vec2 
vec2::unitVector() const 
{   
	vec2 u(*this);
	u.normalize();
	return u;
}


float 
vec2::dot(const vec2 &v) const 
{
	return (x * v.x + y * v.y);
}
		   
		   
float 
vec2::angle(vec2 &v) const 
{
	vec2 aux;
		   
	aux = this->unitVector();
	v.normalize();
		   
	return acosf(aux.dot(v));
}


const vec2 
vec2::lerp (const vec2 &v, float alpha) const {
	vec2 result;

	float ix = x + ((v.x - x) * alpha);
	float iy = y + ((v.y - y) * alpha);

	result.set(ix, iy);

	return (result);
}


void vec2::add (const vec2 &v)
{
	x += v.x;
	y += v.y; 
}


void 
vec2::scale (float a)
{				
	x *= a;
	y *= a;
}
		   

bool 
vec2::equals (const vec2 &v, float tolerance) const {
	return (FloatEqual(x, v.x, tolerance) && FloatEqual(y, v.y, tolerance));
}   
