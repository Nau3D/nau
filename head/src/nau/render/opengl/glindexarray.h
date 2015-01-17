#ifndef GLINDEXARRAY_H
#define GLINDEXARRAY_H

#include <nau/render/indexdata.h>

#include <GL/glew.h>
#include <GL/gl.h>

namespace nau
{
	namespace render
	{
		class GLIndexArray : public IndexData
		{
		protected:
			bool m_IsCompiled;
			GLuint m_GLBuffer;
			//bool compile (VertexData &v);
			void resetCompilationFlag();
			bool isCompiled();
			void bind (void);
			void unbind (void);

		public:
			GLIndexArray(void);

			std::vector<unsigned int>& getIndexData (void);
			void compile();
			void useAdjacency(bool adj);
			bool getAdjacency();

			virtual unsigned int getBufferID();
			void setBuffer(unsigned int id);

			~GLIndexArray(void);

		};
	};
};

#endif //GLVERTEXARRAY_H
