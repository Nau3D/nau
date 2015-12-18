#include "nau/event/interpolatorFactory.h"
#include "nau/event/positionInterpolator.h"

using namespace nau::event_;
using namespace std;

nau::event_::Interpolator* 
InterpolatorFactory::create (std::string type)
{
	if ("PositionInterpolator" == type) {
		return new nau::event_::PositionInterpolator;
	}

	return 0;
}