#ifndef GRID_H
#define GRID_H

#include "nau/geometry/primitive.h"

namespace nau
{
	namespace geometry
	{
		class Grid : public Primitive
		{
		public:
			Grid(void);
			~Grid(void);

			static AttribSet Attribs;
			static AttribSet &GetAttribs() { return Attribs; }

			UINT_PROP(DIVISIONS, 0);
			FLOAT_PROP(LENGTH, 1);

			void build();
			void setPropui(UIntProperty prop, int unsigned value);
			void setPropf(FloatProperty prop, float value);

		protected:

			static bool Init();
			static bool Inited;

			bool m_Built;

			void rebuild();
		};
	};
};
#endif
