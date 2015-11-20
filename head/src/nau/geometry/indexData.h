#ifndef INDEXDATA_H
#define INDEXDATA_H

#include "nau/math/vec4.h"

#include <map>
#include <memory>
#include <vector>

using namespace nau::math;

namespace nau
{
	namespace geometry
	{
		class IndexData
		{
		public:

			static std::shared_ptr<IndexData> Create (std::string &);

			virtual ~IndexData(void);

			void setName(std::string name);

			void offsetIndices (int amount);
			virtual std::shared_ptr<std::vector<unsigned int>> & getIndexData (void);
#ifdef NAU_OPTIX
			/// required for optixPrime: returns indices as ints
			virtual void getIndexDataAsInt(std::vector<int> *);
#endif
			void setIndexData (std::shared_ptr<std::vector<unsigned int>> &);
			/// returns the number of indices
			unsigned int getIndexSize (void);

			void buildAdjacencyList();

			void add (std::shared_ptr<IndexData> &anIndexData);

			virtual void useAdjacency(bool adj) = 0;
			virtual bool getAdjacency() = 0;

			virtual unsigned int getBufferID() = 0;

			virtual void setBuffer(unsigned int id) = 0;

			virtual void resetCompilationFlag() = 0;
			virtual void compile() = 0;
			virtual void bind (void) = 0;
			virtual void unbind (void) = 0;
			virtual bool isCompiled() = 0;

		protected:
			IndexData(void);
			IndexData(std::string &name);
			std::string m_Name;
			
			std::shared_ptr<std::vector<unsigned int>> m_InternalIndexArray;
			std::shared_ptr<std::vector<unsigned int>> m_AdjIndexArray;

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
