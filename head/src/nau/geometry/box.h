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


			static const std::string FloatParamNames[];
			
			typedef enum {COUNT_FLOATPARAMS} FloatParams;
			
			void setParam(unsigned int, float value);
			float getParamf(unsigned int param);
			const std::string &getParamfName(unsigned int i);
			void build();

			virtual unsigned int translate(const std::string &name);

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
