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


			static const std::string FloatParamNames[];
			
			typedef enum {COUNT_FLOATPARAMS} FloatParams;
			
			void setParam(unsigned int, float value);
			float getParamf(unsigned int param);
			const std::string &getParamfName(unsigned int i);
			void build();

			virtual unsigned int translate(const std::string &name);

		private:
			
			std::vector<float> m_Floats;

			// The four corners of the box
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
