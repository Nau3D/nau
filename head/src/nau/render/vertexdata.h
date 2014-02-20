#ifndef VERTEXDATA_H
#define VERTEXDATA_H

#include <nau/math/vec4.h>

#include <map>
#include <vector>

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

			typedef nau::math::vec4 Attr ;
			static std::vector<Attr> NoData;
//			static std::vector<unsigned int> NoIndexData;
		
			static unsigned int const NOLOC = 0;

			static VertexData* create (void);

		public:
			//static const int ArraysCount = 23;
			//static const int AttributesCount = 8;
			//typedef enum {
			//	VERTEX_ARRAY = 0,
			//	NORMAL_ARRAY,
			//	COLOR_ARRAY,
			//	SECONDARY_COLOR_ARRAY,
			//	EDGE_ARRAY,
			//	FOG_ARRAY,
			//	TEXTURE_COORD_ARRAY0,
			//	TEXTURE_COORD_ARRAY1,
			//	TEXTURE_COORD_ARRAY2,
			//	TEXTURE_COORD_ARRAY3,
			//	TEXTURE_COORD_ARRAY4,
			//	TEXTURE_COORD_ARRAY5,
			//	TEXTURE_COORD_ARRAY6,
			//	TEXTURE_COORD_ARRAY7,
			//	CUSTOM_ATTRIBUTE_ARRAY0, // tangent
			//	CUSTOM_ATTRIBUTE_ARRAY1, // triangleID
			//	CUSTOM_ATTRIBUTE_ARRAY2,
			//	CUSTOM_ATTRIBUTE_ARRAY3,
			//	CUSTOM_ATTRIBUTE_ARRAY4,
			//	CUSTOM_ATTRIBUTE_ARRAY5,
			//	CUSTOM_ATTRIBUTE_ARRAY6,
			//	CUSTOM_ATTRIBUTE_ARRAY7,
			//	INDEX_ARRAY
			//} VertexDataType;

			typedef enum {
				DRAW_VERTICES = 0x01,
				DRAW_NORMALS = 0x02,
				DRAW_COLORS = 0x04,
				DRAW_SECONDARY_COLORS = 0x08,
				DRAW_EDGES = 0x10,
				DRAW_FOG = 0x20,
				DRAW_TEXTURE_COORDS = 0x40
			} DrawArrays;
			
		public:
			virtual ~VertexData(void);

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
			void unitize(float min, float max);
			virtual bool compile (void) = 0;
			virtual void resetCompilationFlag() = 0;
			virtual void bind (void) = 0;
			//virtual void bind (unsigned int buffers) = 0;
			virtual void unbind (void) = 0;
			virtual bool isCompiled() = 0;
//			virtual std::vector<unsigned int>& _getReallyIndexData (void) = 0;
			virtual unsigned int getBufferID(unsigned int vertexAttrib) = 0;

		protected:
			VertexData(void);
			
			std::vector<Attr>* m_InternalArrays[MaxAttribs];
//			std::vector<unsigned int>* m_InternalIndexArray;

//			unsigned int m_IndexSize;
			//std::map<VertexDataType, std::vector<vec3>*> m_InternalArrays;
		};
	};
};




#endif //VERTEXDATA_H
