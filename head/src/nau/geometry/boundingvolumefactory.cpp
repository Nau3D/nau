#include "nau/geometry/boundingvolumefactory.h"

#include "nau/geometry/boundingbox.h"

using namespace nau::geometry;

IBoundingVolume* 
BoundingVolumeFactory::create (std::string type)
{
	if (0 == type.compare ("BoundingBox")) {
		return new BoundingBox();
	}
	return 0;
}

