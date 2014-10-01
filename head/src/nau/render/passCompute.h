#ifndef PASS_COMPUTE_H
#define PASS_COMPUTE_H

#include <vector>
#include <string>
#include <map>

#include <nau/render/pass.h>

using namespace nau::render;

namespace nau
{
	namespace render
	{

		class PassCompute : public Pass {

		public:
			PassCompute (const std::string &passName);
			virtual ~PassCompute();

			void eventReceived(const std::string &sender, const std::string &eventType, IEventData *evtData);

			const std::string &getClassName();

			void prepare();
			void restore();
			void doPass();

			void setMaterialName(const std::string &lName,const std::string &mName);
			void setDimension(int dimX, int dimY, int dimZ);
			void setAtomics(int atomicX, int atomicY, int atomicZ);

		protected:
			Material *m_Mat;
			int m_DimX, m_DimY, m_DimZ;
			int m_AtomicX, m_AtomicY, m_AtomicZ;

		};
	};
};
#endif





	
	
