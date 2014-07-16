#ifndef INDEXDATA_H
#define INDEXDATA_H

#include <nau/math/vec4.h>
#include <nau/render/vertexdata.h>

#include <map>
#include <vector>

using namespace nau::math;

namespace nau
{
	namespace render
	{
		class IndexData
		{
		public:

			static std::vector<unsigned int> NoIndexData;
		
			static IndexData* create (void);

			virtual ~IndexData(void);

			void offsetIndices (int amount);
			virtual std::vector<unsigned int>& getIndexData (void);
			void setIndexData (std::vector<unsigned int>* indexData);
			unsigned int getIndexSize (void);

			void buildAdjacencyList();

			int add (IndexData &anIndexData);

			virtual bool compile (VertexData &v) = 0;
			virtual void resetCompilationFlag() = 0;
			virtual void bind (void) = 0;
			virtual void unbind (void) = 0;
			virtual bool isCompiled() = 0;
			virtual void useAdjacency(bool adj) = 0;
			virtual bool getAdjacency() = 0;

			//virtual std::vector<unsigned int>& _getReallyIndexData (void) = 0;
			virtual unsigned int getBufferID() = 0;

		protected:
			IndexData(void);
			
			std::vector<unsigned int>* m_InternalIndexArray;
			std::vector<unsigned int> m_AdjIndexArray;
			//unsigned int m_IndexSize;

			bool m_UseAdjacency;

			struct HalfEdge {
				unsigned int vertex;
				struct HalfEdge *next;
				struct HalfEdge *twin;
			};

		};
	};
};




#endif //INDEXDATA_H
