#ifndef BBOX_H
#define BBOX_H


#include <nau/geometry/primitive.h>

namespace nau
{
	namespace geometry
	{
		class BBox : public Primitive
		{
		public:
			BBox(void);
			~BBox(void);


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


		};
	};
};
#endif
