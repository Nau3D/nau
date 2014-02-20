#include <nau/geometry/primitive.h>

#include <nau/geometry/box.h>
#include <nau/geometry/axis.h>
#include <nau/geometry/bbox.h>
#include <nau/geometry/sphere.h>


using namespace nau::geometry;

const std::string Primitive::NoParam = "";

//Primitive *
//Primitive::Create(const std::string &type) 
//{
//	if ("Box" == type)
//		return new Box();
//	else if ("Axis" == type)
//		return new Axis();
//	else if ("BoundingBox" == type)
//		return new BBox();
//	else if ("Sphere" == type)
//		return new Sphere();
//	else
//		return new Box();
//}


const std::string &
Primitive::getParamfName(unsigned int i) 
{
	return Primitive::NoParam;
}