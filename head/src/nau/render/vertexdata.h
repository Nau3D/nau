#ifndef VERTEXDATA_H
#define VERTEXDATA_H


#include "nau/math/vec3.h"
#include "nau/math/vec4.h"

#include <map>
#include <vector>
#include <string>

using namespace nau::math;

namespace nau
{
	namespace render
	{
		class VertexData
		{
		public:

			static const int MaxAttribs = 16;

			static  const std::string Syntax[]; 
			static unsigned int getAttribIndex(std::string);

			// A vertex attribute is a vec4
			typedef nau::math::vec4 Attr ;
			static std::vector<Attr> NoData;
		
			static unsigned int const NOLOC = 0;

			static VertexData* create (std::string name);

			virtual ~VertexData(void);

			void setName(std::string &name);

			virtual int getNumberOfVertices() = 0;

			//std::vector<Attr>& getDataOf (VertexDataType type);
			std::vector<Attr>& getDataOf (unsigned int type);
			//void setDataFor (VertexDataType type, 
			//	             std::vector<Attr>* dataArray);
			void setDataFor (unsigned int index, 
				             std::vector<Attr>* dataArray);

//			void offsetIndices (int amount);
//			virtual std::vector<unsigned int>& getIndexData (void);
//			void setIndexData (std::vector<unsigned int>* indexData);
//			unsigned int getIndexSize (void);
			int add (VertexData &aVertexData);

			virtual void prepareTriangleIDs(unsigned int sceneObjID, 
				                                       unsigned int primitiveOffset, 
													   std::vector<unsigned int> *index) = 0;
//			virtual void prepareTangents() = 0;
			virtual void appendVertex(unsigned int i) = 0;

			//virtual std::vector<Attr>& getAttributeDataOf (VertexDataType type) = 0;
			//virtual std::vector<Attr>& getAttributeDataOf (unsigned int type) = 0;
			//virtual void setAttributeDataFor (VertexDataType type, 
			//								  std::vector<Attr>* dataArray, 
			//								  int location) = 0;
			virtual void setAttributeDataFor (unsigned int type, 
											  std::vector<Attr>* dataArray, 
											  int location = -1) = 0;
			//virtual void setAttributeLocationFor (VertexDataType type, int location) = 0;
			virtual void setAttributeLocationFor (unsigned int type, int location) = 0;
			void unitize(vec3 &vCenter, vec3 &vMin, vec3 &vMax);
			virtual bool compile(void) = 0;
//			virtual std::vector<unsigned int>& _getReallyIndexData (void) = 0;
			virtual unsigned int getBufferID(unsigned int vertexAttrib) = 0;

			virtual void resetCompilationFlag() = 0;
			virtual void bind (void) = 0;
			virtual void unbind (void) = 0;
			virtual bool isCompiled() = 0;

			virtual void setBuffer(unsigned int type, int bufferID) = 0;

		protected:
			VertexData(void);
			
			std::vector<Attr>* m_InternalArrays[MaxAttribs];
			std::string m_Name;
//			std::vector<unsigned int>* m_InternalIndexArray;

//			unsigned int m_IndexSize;
			//std::map<VertexDataType, std::vector<vec3>*> m_InternalArrays;
		};
	};
};




#endif //VERTEXDATA_H
