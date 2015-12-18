#ifndef SQUARE_H
#define SQUARE_H


#include "nau/geometry/primitive.h"

namespace nau
{
	namespace geometry
	{
		class Square : public Primitive
		{
		public:
			friend class nau::resource::ResourceManager;

			~Square(void);

			void build();

		private:
			
			std::vector<float> m_Floats;

			// The four corners of the square
			enum {
				TOP_LEFT = 0,
				TOP_RIGHT,
				BOTTOM_RIGHT,
				BOTTOM_LEFT
			};

		protected:
			Square(void);


		};
	};
};
#endif
