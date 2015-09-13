#ifndef GLVERTEXARRAY_H
#define GLVERTEXARRAY_H

#include "nau/geometry/vertexData.h"

#include <GL/glew.h>
#include <GL/gl.h>

using namespace nau::geometry;

namespace nau
{
	namespace render
	{
		class GLVertexArray : public VertexData
		{
		private:
			GLuint m_GLBuffers[VertexData::MaxAttribs+1];
			GLuint m_AttributesLocations[VertexData::MaxAttribs];
			bool m_IsCompiled;
			//static unsigned int m_OpenGLOwnAttribs;

		public:
			GLVertexArray(void);
			~GLVertexArray(void);

			void setAttributeDataFor (unsigned int type, 
				                      std::vector<VertexData::Attr>* dataArray, 
									  int location = -1);
			void setAttributeLocationFor (unsigned int type, int location);

			virtual void prepareTriangleIDs(unsigned int sceneObjID, 
													   unsigned int primitiveOffset, 
													   std::vector<unsigned int> *index);

			virtual void appendVertex(unsigned int i);

			virtual unsigned int getNumberOfVertices();

			virtual bool compile (void);
			virtual unsigned int getBufferID(unsigned int vertexAttrib);

			virtual void bind (void);
			virtual void unbind (void);
			virtual bool isCompiled();
			virtual void resetCompilationFlag();

			void setBuffer(unsigned int type, int bufferID);

			//void setAttributeLocationFor (VertexDataType type, int location);
//			virtual void prepareTangents();
			//virtual void bind (unsigned int buffers);
//			virtual std::vector<unsigned int>& _getReallyIndexData (void);
//			static void setCore(bool flag);

			//std::vector<VertexData::Attr>& getAttributeDataOf (VertexDataType type);
			//std::vector<VertexData::Attr>& getAttributeDataOf (unsigned int type);
			//std::vector<unsigned int>& getIndexData (void);
			//void setAttributeDataFor (VertexDataType type, 
			//	                      std::vector<VertexData::Attr>* dataArray, 
			//						  int location);

		protected:
			//void setGLArray (unsigned int type, float* pointer);
			//GLenum translate (unsigned int type);
			//void setGLArray (VertexDataType type, float* pointer);
			//GLenum translate (VertexDataType type);
		};
	};
};

#endif //GLVERTEXARRAY_H
