#include <nau/util/tree.h>

#include <assert.h>

using namespace nau::util;

Tree::Tree() {

}


void 
Tree::clear() {

	t.clear();
}


void 
Tree::appendItem(std::string key, std::string value) {

	t.push_back(std::pair<std::string, Element *>(key, new Element(value)));
}


Tree *
Tree::appendBranch(std::string key, std::string value) {

	Element *e = new Element();
	Tree *tt = new Tree();

	e->value = value;
	e->values = tt;
	t.push_back(std::pair<std::string, Element *>(key, e));
	return tt;
}


void 
Tree::getKeys(std::vector<std::string> *keys) {

	for (auto k : t) {
		keys->push_back(k.first);
	}
}


Tree *
Tree::getBranch(std::string key, std::string value) {

	for (auto b : t) {
		if (b.first == key && b.second->value == value && b.second->values != NULL)
			return b.second->values;
	}
	return NULL;
}


std::string 
Tree::getValue(std::string key) {

	for (auto b : t) {
		if (b.first == key)
			return b.second->value;
	}
	return std::string("");
}


bool 
Tree::isBranch(std::string key) {

	for (auto b : t) {
		if (b.first == key && b.second->values != NULL)
			return true;
	}
	return false;
}


bool 
Tree::isBranchEmpty(std::string key) {

	for (auto b : t) {
		if (b.first == key && b.second->values != NULL) {
			if (b.second->values->t.size() != 0)
				return false;
			else
				return true;
		}
	}
	return false;
}

	
bool
Tree::hasKey(std::string key) {

	for (auto b : t) {
		if (b.first == key)
			return true;
	}
	return false;
}


bool
Tree::hasElement(std::string key, std::string value) {

	for (auto b : t) {
		if (b.first == key && b.second->value == value)
			return true;
	}
	return false;
}


size_t 
Tree::getElementCount() {

	return t.size();
}


std::string &
Tree::getKey(unsigned int i) {

	assert(i < t.size());
	return t[i].first;
}


std::string &
Tree::getValue(unsigned int i) {

	assert(i < t.size());
	return t[i].second->value;
}


Tree *
Tree::getBranch(unsigned int i) {

	assert(i < t.size());
	return t[i].second->values;
}
