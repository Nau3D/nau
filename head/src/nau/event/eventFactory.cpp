#include "nau/event/eventFactory.h"
#include "nau/event/eventVec3.h"
#include "nau/event/eventVec4.h"
#include "nau/event/eventString.h"
#include "nau/event/eventFloat.h"
#include "nau/event/eventInt.h"
#include "nau/event/eventCameraOrientation.h"
#include "nau/event/eventCameraMotion.h"
#include <assert.h>

using namespace nau::event_;
using namespace std;

std::shared_ptr<IEventData>
EventFactory::Create (std::string type)
{
	if ("String" == type) {
		return std::shared_ptr<IEventData>(new nau::event_::EventString);
	}

	if ("Vec3" == type) {
		return std::shared_ptr<IEventData>(new nau::event_::EventVec3);
	}

	if ("Vec4" == type) {
		return std::shared_ptr<IEventData>(new nau::event_::EventVec4);
	}

	if ("Float" == type) {
		return std::shared_ptr<IEventData>(new nau::event_::EventFloat);
	}

	if ("Camera Orientation" == type) {
		return std::shared_ptr<IEventData>(new nau::event_::EventCameraOrientation);
	}

	if ("Camera Motion" == type) {
		return std::shared_ptr<IEventData>(new nau::event_::EventCameraMotion);
	}
	if ("Int" == type) {
		return std::shared_ptr<IEventData>(new nau::event_::EventInt);
	}

	assert("EventFactory: invalid type");
	return std::shared_ptr<IEventData>();
}