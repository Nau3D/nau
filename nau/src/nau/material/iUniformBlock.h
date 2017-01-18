#ifndef IUNIFORMBLOCK_H
#define IUNIFORMBLOCK_H

#include "nau/enums.h"
#include "nau/material/iBuffer.h"

#include <string>
#include <map>

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
		class UniformBlockManager;

		class IUniformBlock 
		{
			friend class UniformBlockManager;

		protected:
			IUniformBlock() {};

		public:
		
			nau_API static IUniformBlock *Create(std::string &name, unsigned int size);

			nau_API virtual ~IUniformBlock() {};

			nau_API virtual void init(std::string &name, unsigned int size) = 0;
			nau_API virtual void setBindingIndex(unsigned int i) = 0;
			nau_API virtual unsigned int getBindingIndex() = 0;
			nau_API virtual void addUniform(std::string &name, Enums::DataType type,
				unsigned int offset, unsigned int size = 0, 
				unsigned int arrayStride = 0) = 0;
			nau_API virtual void setUniform(std::string &name, void *value) = 0;
			nau_API virtual void setBlock(void *value) = 0;
			nau_API virtual void sendBlockData() = 0;
			nau_API virtual void useBlock() = 0;
			nau_API virtual void getUniformNames(std::vector<std::string> *s) = 0;

			nau_API virtual unsigned int getSize() = 0;
			nau_API virtual bool hasUniform(std::string &name) = 0;
			nau_API virtual Enums::DataType getUniformType(std::string name) = 0;
			nau_API virtual IBuffer *getBuffer() = 0;
		};
	};
};

#endif