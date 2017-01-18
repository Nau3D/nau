#ifndef POSITIONINTERPOLATOR_H
#define POSITIONINTERPOLATOR_H

#include "nau/event/eventVec4.h"
#include "nau/event/iInterpolator.h"
#include "nau/math/vec4.h"

#include <memory>

namespace nau
{
	namespace event_
	{
			class PositionInterpolator: public Interpolator
			{
			private:
				float fraction;

			public:
				PositionInterpolator(const PositionInterpolator &c);
				PositionInterpolator(void);
				~PositionInterpolator(void);

				void removePositionListener(void);
				void addPositionListener(void);
				void eventReceived(const std::string &sender, const std::string &eventType, 
					const std::shared_ptr<IEventData> &evt);
			};
	};
};
#endif
