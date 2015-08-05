#ifndef INDEXDATA_H
#define INDEXDATA_H

#include "nau/math/vec4.h"

#include <map>
#include <vector>

using namespace nau::math;

namespace nau
{
	namespace geometry
	{
		class IndexData
		{
		public:

			static std::vector<unsigned int> NoIndexData;
		
			static IndexData* create (std::string);

			virtual ~IndexData(void);

			void setName(std::string name);

			void offsetIndices (int amount);
			virtual std::vector<unsigned int>& getIndexData (void);
#ifdef NAU_OPTIX_PRIME
			/// required for optixPrime: returns indices as ints
			virtual std::vector<int>* getIndexDataAsInt(void);
#endif
			void setIndexData (std::vector<unsigned int>* indexData);
			/// returns the number of indices
			unsigned int getIndexSize (void);

			void buildAdjacencyList();

			int add (IndexData &anIndexData);

			virtual void useAdjacency(bool adj) = 0;
			virtual bool getAdjacency() = 0;

			//virtual std::vector<unsigned int>& _getReallyIndexData (void) = 0;
			virtual unsigned int getBufferID() = 0;

			virtual void setBuffer(unsigned int id) = 0;

			virtual void resetCompilationFlag() = 0;
			virtual void compile() = 0;
			virtual void bind (void) = 0;
			virtual void unbind (void) = 0;
			virtual bool isCompiled() = 0;

		protected:
			IndexData(void);
			std::string m_Name;
			
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
