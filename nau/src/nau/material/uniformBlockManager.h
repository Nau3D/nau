#ifndef UNIFORMBLOCKMANAGER_H
#define UNIFORMBLOCKMANAGER_H


#include "nau/enums.h"
#include "nau/material/iUniformBlock.h"

#include <string>
#include <map>

#define UNIFORMBLOCKMANAGER UniformBlockManager::GetInstance()

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif


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
			nau_API static UniformBlockManager *GetInstance();
			nau_API static void DeleteInstance();
			nau_API ~UniformBlockManager();

			nau_API void clear();
			nau_API void addBlock(std::string &name, unsigned int size);
			nau_API IUniformBlock *getBlock(const std::string &name);
			nau_API bool hasBlock(std::string &name);

			nau_API unsigned int getCurrentBindingIndex();

		};
	};
};

#endif