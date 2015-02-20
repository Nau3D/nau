#include "nau/math/transformfactory.h"

#include "nau/math/simpletransform.h"

using namespace nau::math;

ITransform* 
TransformFactory::create (std::string type)
{
	if (0 == type.compare ("SimpleTransform")) {
		return new SimpleTransform;
	}
	return 0;
}
