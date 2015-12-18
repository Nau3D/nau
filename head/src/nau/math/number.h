#ifndef NAU_NUMBER_H
#define NAU_NUMBER_H

#include "nau/math/data.h"

namespace nau {

	namespace math {

		template <class T>
		class Number : public Data {

		protected:
			T m_Number;

		public:

			Number() {
				m_Number = T(0);
			}

			Number(T num) {
				m_Number = num;
			}

			Data *clone() {
				return new Number(m_Number);
			}

			/// returns a raw pointer to the class data
			void *getPtr() {
				return &m_Number;
			}

			T operator=(T arg) {
				m_Number = arg;
			}

			bool operator<(const T& T) const {
				if (m_Number < a.m_Number)
					return true;
				else
					return false;
			}

			bool operator>(const T& T) const {
				if (m_Number > a.m_Number)
					return true;
				else
					return false;
			}

			T getNumber() {
				return m_Number;
			}

			operator T() {
				return m_Number;
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