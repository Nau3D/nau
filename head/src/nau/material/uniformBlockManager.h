#ifndef UNIFORMBLOCKMANAGER_H
#define UNIFORMBLOCKMANAGER_H


#include "nau/enums.h"
#include "nau/material/iUniformBlock.h"

#include <string>
#include <map>

#define UNIFORMBLOCKMANAGER UniformBlockManager::GetInstance()

namespace nau
{
	namespace material
	{
		class UniformBlockManager {

		protected:
			UniformBlockManager();

			static UniformBlockManager *Instance;
			std::map<std::string, IUniformBlock *> m_Blocks;

		public:
			static UniformBlockManager *GetInstance();
			~UniformBlockManager();

			void clear();
			void addBlock(std::string &name, unsigned int size);
			IUniformBlock *getBlock(std::string &name);
			bool hasBlock(std::string &name);

			unsigned int getCurrentBindingIndex();

		};
	};
};

#endif