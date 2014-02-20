#include <nau/math/plane.h>

using namespace nau::math;

plane::plane(void) 
	: normal()
{
}

plane::~plane()
{
}

void
plane::setCoefficients (float a, float b, float c, float d)
{
	normal.set (a, b, c);

	float l = normal.length ();

	normal /= l;

	this->d = d/l;
}

float
plane::distance(vec3 &v)
{
	return (d + normal.dot (v));
}

const vec3& 
plane::getNormal (void)
{
	return normal;
}
