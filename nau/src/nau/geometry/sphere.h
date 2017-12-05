#ifndef SPHERE_H
#define SPHERE_H


#include "nau/geometry/primitive.h"

namespace nau
{
	namespace geometry
	{
		class Sphere : public Primitive
		{
		public:

			Sphere(void);			
			~Sphere(void);

			std::string getClassName();

			static AttribSet Attribs;
			static AttribSet &GetAttribs() { return Attribs; }

			UINT_PROP(SLICES, 0);
			UINT_PROP(STACKS, 1);

			void build();
			void setPropui(UIntProperty prop, int unsigned value);

		protected:

			//std::vector<float> m_Floats;
			static bool Init();
			static bool Inited;

			bool m_Built;

			void rebuild();

		};
	};
};
#endif
