#include "nau/material/uniformBlockManager.h"

using namespace nau::material;

UniformBlockManager *UniformBlockManager::Instance = NULL;


UniformBlockManager *
UniformBlockManager::GetInstance() {

	if (Instance == NULL)
		Instance = new UniformBlockManager;

	return Instance;
}

void 
UniformBlockManager::DeleteInstance() {

	if (Instance) {
		delete Instance;
		Instance = NULL;
	}
}


UniformBlockManager::~UniformBlockManager() {

	clear();
}


UniformBlockManager::UniformBlockManager() {

	//while (!m_Blocks.empty()){
	//	delete((*m_Blocks.begin()).second);
	//	m_Blocks.erase(m_Blocks.begin());
	//}
}


void 
UniformBlockManager::clear() {

	while (!m_Blocks.empty()) {
		delete((*m_Blocks.begin()).second);
		m_Blocks.erase(m_Blocks.begin());
	}
}


void
UniformBlockManager::addBlock(std::string &name, unsigned int size) {

	m_Blocks[name] = IUniformBlock::Create(name, size);
}


IUniformBlock *
UniformBlockManager::getBlock(const std::string &name) {

	if (m_Blocks.count(name))
		return m_Blocks[name];
	else
		return NULL;
}
			

bool 
UniformBlockManager::hasBlock(std::string &name) {

	if (m_Blocks.count(name))
		return true;
	else
		return false;

}


unsigned int
UniformBlockManager::getCurrentBindingIndex() {

	return (unsigned int)m_Blocks.size();
}