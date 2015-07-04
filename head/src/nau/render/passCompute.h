#ifndef PASS_COMPUTE_H
#define PASS_COMPUTE_H

#include <map>
#include <string>
#include <vector>


#include "nau/material/material.h"
#include "nau/render/pass.h"

using namespace nau::render;

namespace nau
{
	namespace render
	{

		class PassCompute : public Pass {

		public:

			UINT_PROP(DIM_X, 101);
			UINT_PROP(DIM_Y, 102);
			UINT_PROP(DIM_Z, 103);

			PassCompute (const std::string &passName);
			virtual ~PassCompute();

			void eventReceived(const std::string &sender, const std::string &eventType, IEventData *evtData);

			const std::string &getClassName();

			void prepare();
			void restore();
			void doPass();

			void setMaterialName(const std::string &lName,const std::string &mName);
			Material *getMaterial();

			void setDimension(int dimX, int dimY, int dimZ);
			void setDimFromBuffer(IBuffer  *buffNameX, unsigned int offX,
				IBuffer  *buffNameY, unsigned int offY,
				IBuffer  *buffNameZ, unsigned int offZ);

		protected:

			static bool Init();
			static bool Inited;

			Material *m_Mat;
			IBuffer  *m_BufferX, *m_BufferY, *m_BufferZ;
			unsigned int m_OffsetX, m_OffsetY, m_OffsetZ;

		};
	};
};
#endif





	
	
