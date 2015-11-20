#ifndef BBOX_H
#define BBOX_H


#include "nau/geometry/primitive.h"

namespace nau
{
	namespace geometry
	{
		class BBox : public Primitive
		{
		public:
			BBox(void);
			~BBox(void);


			void build();

		private:
			
			std::vector<float> m_Floats;
		};
	};
};
#endif
