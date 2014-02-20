#ifndef BIMAP_H
#define BIMAP_H

#include <map>

namespace nau {

	class BiMap {

	public:
		
		std::map<int,int> left, right;

		BiMap() {}
		~BiMap() {}

		void add(int l,int r) {
		
			left[l] = r;
			right[r] = l;
		}

		int getLeft(int r) {
		
			return (right[r]);
		}

		int getRight(int l) {
		
			return (left[r]);
		}
	}
};


#endif
