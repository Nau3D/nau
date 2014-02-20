#ifndef INTERPOLATORFACTORY_H
#define INTERPOLATORFACTORY_H

#include <nau/event/interpolator.h>
#include <string>

namespace nau
{
	namespace event_
	{
		class InterpolatorFactory
		{
		public:
			static nau::event_::Interpolator* create (std::string type);
		private:
			InterpolatorFactory(void) {};
			~InterpolatorFactory(void) {};
		};
	};
};
#endif