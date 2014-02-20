#ifndef SPHERE_H
#define SPHERE_H


#include <nau/geometry/primitive.h>

namespace nau
{
	namespace geometry
	{
		class Sphere : public Primitive
		{
		public:
			Sphere(void);			
			~Sphere(void);


			static const std::string FloatParamNames[];
			
			typedef enum {SLICES, STACKS, COUNT_FLOATPARAMS} FloatParams;
			
			void setParam(unsigned int, float value);
			float getParamf(unsigned int param);
			const std::string &getParamfName(unsigned int i);
			void build();

			virtual unsigned int translate(const std::string &name);

		private:

			std::vector<float> m_Floats;

		};
	};
};
#endif
