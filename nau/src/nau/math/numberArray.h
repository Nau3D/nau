#ifndef NAU_NUMBERARRAY_H
#define NAU_NUMBERARRAY_H


#include "nau/math/data.h"

#include <vector>

namespace nau {

	namespace math {

		template <class T>
		class NumberArray : public Data {

		protected:
			std::vector<T> m_Numbers;

		public:

			NumberArray() {
				m_Numbers.clear();
			};

			NumberArray(const NumberArray &na) {
				m_Numbers = na.m_Numbers;;
			};

			NumberArray(std::vector<T> &nums) {
				m_Numbers = nums;
			};

			~NumberArray() {};

			Data *clone() {
				return new NumberArray(*this);
			};

			/// returns a raw pointer to the class data
			void *getPtr() {
				if (m_Numbers.size())
					return &(m_Numbers[0]);
				else
					return &m_Numbers;
			};

			const std::vector<T> &getArray() const {
				return m_Numbers;
			};

			void operator = (const NumberArray &v) {
				this->m_Numbers = v.m_Numbers;
			};

			void append(T t) {
				m_Numbers.push_back(t);
			};

			int size() {
				return (int)m_Numbers.size();
			};

			void clear() {
				m_Numbers.clear();
			};

		};
		typedef NumberArray<int> NauIntArray;
	};
};

#endif
