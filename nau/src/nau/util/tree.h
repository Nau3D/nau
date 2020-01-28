#ifndef TREE
#define TREE

#include <map>
#include <vector>
#include <string>

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

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

			void clear();
		};

		class Tree {

		typedef std::vector<std::pair<std::string, Element *>> tree;

		protected:
			tree t;

		public:

			Tree();

			nau_API Tree *getTree();
			nau_API void clear();

			nau_API void appendItem(std::string key, std::string value);
			nau_API Tree *appendBranch(std::string key, std::string value = "");

			nau_API void getKeys(std::vector<std::string> *keys);

			nau_API Tree *getBranch(std::string key,std::string value = "");
			nau_API std::string getValue(std::string key);

			nau_API bool isBranch(std::string key);
			nau_API bool isBranchEmpty(std::string key);

			nau_API bool hasKey(std::string key);
			nau_API bool hasElement(std::string key, std::string value);

			nau_API size_t getElementCount();
			nau_API std::string &getKey(unsigned int i);
			nau_API std::string &getValue(unsigned int i);
			nau_API Tree *getBranch(unsigned int i);

		};
	};
};

#endif