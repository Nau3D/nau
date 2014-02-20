#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include <nau/geometry/mesh.h>

namespace nau
{
	namespace geometry
	{
		class Primitive : public Mesh
		{
		protected:
			//Primitive(void): Mesh() {};
			//~Primitive(void);

		protected:

		public:
			//static Primitive *Create(const std::string &type);
			//static Primitive *Create(const std::string &type, int a, int b);

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
