#ifndef TRANSFORMFACTORY_H
#define TRANSFORMFACTORY_H

#include <string>

#include <nau/math/itransform.h>

namespace nau
{
	namespace math
	{
		class TransformFactory
		{
		public:
			static ITransform* create (std::string type);
		private:
			TransformFactory(void);
			~TransformFactory(void);
		};
	};
};

#endif
