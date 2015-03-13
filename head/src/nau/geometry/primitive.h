#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "nau/geometry/mesh.h"

namespace nau
{
	namespace geometry
	{
		class Primitive : public Mesh
		{
		public:

			virtual void setParam(unsigned int, float value) {};
			virtual float getParamf(unsigned int) = 0;

			virtual unsigned int translate(const std::string &name) = 0;
			virtual const std::string &getParamfName(unsigned int i);
			virtual void build() = 0;

			static const std::string NoParam;
		};
	};
};
#endif
