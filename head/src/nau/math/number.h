#ifndef NAU_NUMBER_H
#define NAU_NUMBER_H

#include "nau/math/data.h"

namespace nau {

	namespace math {

		template <class T>
		class Number : public Data {

		protected:
			T number;

		public:

			Number(T num) {
				number = num;
			}

			T operator=(T arg) {
				number = arg;
			}

			operator T() {
				return number;
			}
		};

		typedef Number<int> NauInt;
		typedef Number<unsigned int> NauUInt;
		typedef Number<float> NauFloat;
		typedef Number<double> NauDouble;
		typedef Number<short> NauShort;
		typedef Number<char> NauByte;
		typedef Number<unsigned char> NauUByte;
		typedef Number<short> NauShort;
		typedef Number<unsigned short> NauUShort;
		
	};
};

#endif