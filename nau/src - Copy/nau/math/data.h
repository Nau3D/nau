#ifndef NAU_DATA_H
#define NAU_DATA_H


namespace nau {

	namespace math {

		class Data {
		protected:
			Data() {};
		public:
			virtual ~Data() {};
			virtual void *getPtr() = 0;
			virtual Data *clone() = 0;
		};
	};
};

#endif