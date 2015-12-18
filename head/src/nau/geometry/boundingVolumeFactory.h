#ifndef BOUNDINGVOLUMEFACTORY_H
#define BOUNDINGVOLUMEFACTORY_H

#include "nau/geometry/iBoundingVolume.h"

namespace nau
{
	namespace geometry
	{
		class BoundingVolumeFactory
		{
		public:
			static IBoundingVolume* create (std::string type);

		private:
			BoundingVolumeFactory(void);
			~BoundingVolumeFactory(void);
		};
	};
};

#endif
