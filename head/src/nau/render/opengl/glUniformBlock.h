#ifndef GL_UNIFORMBLOCK_H
#define GL_UNIFORMBLOCK_H

#include "nau/enums.h"

#include "nau/material/iUniformBlock.h"


#include <string>
#include <map>

using namespace nau::material;

namespace nau
{
	namespace render
	{
		class GLUniformBlock: public nau::material::IUniformBlock
		{

		private:
			/// stores information for block uniforms
			typedef struct blockUniforms {
				Enums::DataType  type;
				unsigned int offset;
				unsigned int  size;
				unsigned int  arrayStride;
			} blockUniform;

			/// size of the uniform block
			int m_Size;
			/// buffer bound to the index point
			IBuffer *m_Buffer;
			/// uniforms information
			std::map<std::string, blockUniform > m_Uniforms;
			/// stores uniform values locally 
			void *m_LocalData;
			//
			unsigned int m_BindingIndex;
			// true if setUniform or setBlock have been called after last call to sendBlockData or useBlock
			bool m_BlockChanged;

			std::string m_Name;

			// largest possible value for a GLSL data type
			char m_Std140Value[128];

			GLUniformBlock();

		public:
		
			~GLUniformBlock();

			GLUniformBlock(std::string &name, unsigned int size);
			void init(std::string &name, unsigned int size);
			void setBindingIndex(unsigned int i);
			unsigned int getBindingIndex();
			void addUniform(std::string &name, Enums::DataType type, unsigned int offset, unsigned int size = 0, unsigned int arrayStride = 0);
			void setUniform(std::string &name, void *value);
			void setBlock(void *value);
			void sendBlockData();
			void useBlock();

			void getUniformNames(std::vector<std::string> *s);
			unsigned int getSize();
			bool hasUniform(std::string &name);
			Enums::DataType getUniformType(std::string name);
			IBuffer *getBuffer();
		};
	};
};

#endif