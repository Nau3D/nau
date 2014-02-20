#ifndef GLVERTEXARRAY_H
#define GLVERTEXARRAY_H

#include <nau/render/vertexdata.h>

#include <GL/glew.h>
#include <GL/gl.h>

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
			static unsigned int m_OpenGLOwnAttribs;

		public:
			GLVertexArray(void);

			static void setCore(bool flag);

			//std::vector<VertexData::Attr>& getAttributeDataOf (VertexDataType type);
			//std::vector<VertexData::Attr>& getAttributeDataOf (unsigned int type);
			//std::vector<unsigned int>& getIndexData (void);
			//void setAttributeDataFor (VertexDataType type, 
			//	                      std::vector<VertexData::Attr>* dataArray, 
			//						  int location);
			void setAttributeDataFor (unsigned int type, 
				                      std::vector<VertexData::Attr>* dataArray, 
									  int location = -1);
			//void setAttributeLocationFor (VertexDataType type, int location);
			void setAttributeLocationFor (unsigned int type, int location);

			virtual void prepareTriangleIDs(unsigned int sceneObjID, 
													   unsigned int primitiveOffset, 
													   std::vector<unsigned int> *index);

//			virtual void prepareTangents();
			virtual void appendVertex(unsigned int i);

			virtual bool compile (void);
			virtual void bind (void);
			//virtual void bind (unsigned int buffers);
			virtual void unbind (void);
			virtual bool isCompiled();
			virtual void resetCompilationFlag();
//			virtual std::vector<unsigned int>& _getReallyIndexData (void);
			virtual unsigned int getBufferID(unsigned int vertexAttrib);
		public:
			~GLVertexArray(void);

		private:
			//void setGLArray (VertexDataType type, float* pointer);
			void setGLArray (unsigned int type, float* pointer);
//			GLenum translate (VertexDataType type);
			GLenum translate (unsigned int type);
		};
	};
};

#endif //GLVERTEXARRAY_H
