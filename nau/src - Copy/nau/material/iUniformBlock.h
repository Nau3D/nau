#ifndef IUNIFORMBLOCK_H
#define IUNIFORMBLOCK_H

#include "nau/enums.h"
#include "nau/material/iBuffer.h"

#include <string>
#include <map>

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
		
			static IUniformBlock *Create(std::string &name, unsigned int size);

			 virtual ~IUniformBlock() {};

			virtual void init(std::string &name, unsigned int size) = 0;
			virtual void setBindingIndex(unsigned int i) = 0;
			virtual unsigned int getBindingIndex() = 0;
			virtual void addUniform(std::string &name, Enums::DataType type, 
				unsigned int offset, unsigned int size = 0, 
				unsigned int arrayStride = 0) = 0;
			virtual void setUniform(std::string &name, void *value) = 0;
			virtual void setBlock(void *value) = 0;
			virtual void sendBlockData() = 0;
			virtual void useBlock() = 0;
			virtual void getUniformNames(std::vector<std::string> *s) = 0;

			virtual unsigned int getSize() = 0;
			virtual bool hasUniform(std::string &name) = 0;
			virtual Enums::DataType getUniformType(std::string name) = 0;
			virtual IBuffer *getBuffer() = 0;
		};
	};
};

#endif