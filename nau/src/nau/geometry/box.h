#ifndef BOX_H
#define BOX_H


#include "nau/geometry/primitive.h"

namespace nau
{
	namespace geometry
	{
		class Box : public Primitive
		{
		public:
			friend class nau::resource::ResourceManager;

			~Box(void);

			std::string getClassName();

			void build();

		private:
			
			std::vector<float> m_Floats;

			// The eight corners of the box
			enum {
				TOP_LEFT = 0,
				TOP_RIGHT,
				BOTTOM_RIGHT,
				BOTTOM_LEFT
			};
			enum {
				FACE_FRONT,
				FACE_LEFT = 4,
				FACE_BACK = 8,
				FACE_RIGHT = 12,
				FACE_TOP = 16,
				FACE_BOTTOM = 20
			};

		protected:
			Box(void);


		};
	};
};
#endif
