#ifndef POSITIONINTERPOLATOR_H
#define POSITIONINTERPOLATOR_H

#include <nau/event/eventvec4.h>
#include <nau/event/interpolator.h>
#include <nau/math/vec4.h>


namespace nau
{
	namespace event_
	{
			class PositionInterpolator: public Interpolator
			{
			private:
				float fraction;
				nau::event_::EventVec4 e;

			public:
				PositionInterpolator(const PositionInterpolator &c);
				PositionInterpolator(void);
				~PositionInterpolator(void);

				void removePositionListener(void);
				void addPositionListener(void);
				void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);
			};
	};
};
#endif