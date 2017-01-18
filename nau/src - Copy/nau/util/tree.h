#ifndef TREE
#define TREE

#include <map>
#include <vector>

namespace nau {

	namespace util {

		class Tree;

		class Element {

			friend class Tree;

		protected:

			std::string value;
			Tree *values;

			Element(std::string v) {
				value = v;
				values = NULL;
			};

			Element() {
				value = "";
				values = NULL;
			};
		};

		class Tree {

		typedef std::vector<std::pair<std::string, Element *>> tree;

		protected:
			tree t;

		public:

			Tree();

			Tree *getTree();
			void clear();

			void appendItem(std::string key, std::string value);
			Tree *appendBranch(std::string key, std::string value = "");

			void getKeys(std::vector<std::string> *keys);

			Tree *getBranch(std::string key,std::string value = "");
			std::string getValue(std::string key);

			bool isBranch(std::string key);
			bool isBranchEmpty(std::string key);

			bool hasKey(std::string key);
			bool hasElement(std::string key, std::string value);

			size_t getElementCount();
			std::string &getKey(unsigned int i);
			std::string &getValue(unsigned int i);
			Tree *getBranch(unsigned int i);

		};
	};
};

#endif