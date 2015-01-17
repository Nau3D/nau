#include <nau/geometry/primitive.h>

#include <nau/geometry/box.h>
#include <nau/geometry/axis.h>
#include <nau/geometry/bbox.h>
#include <nau/geometry/sphere.h>


using namespace nau::geometry;

const std::string Primitive::NoParam = "";

const std::string &
Primitive::getParamfName(unsigned int i) 
{
	return Primitive::NoParam;
}