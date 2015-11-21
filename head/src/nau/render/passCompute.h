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
			friend class PassFactory;

		public:

			UINT_PROP(DIM_X, 101);
			UINT_PROP(DIM_Y, 102);
			UINT_PROP(DIM_Z, 103);

			virtual ~PassCompute();

			static std::shared_ptr<Pass> Create(const std::string &name);

			void eventReceived(const std::string &sender, const std::string &eventType, 
				const std::shared_ptr<IEventData> &evt);

			const std::string &getClassName();

			void prepare();
			void restore();
			void doPass();

			void setMaterialName(const std::string &lName,const std::string &mName);
			std::shared_ptr<Material> &getMaterial();

			void setDimension(unsigned int dimX, unsigned int dimY, unsigned int dimZ);
			void setDimFromBuffer(IBuffer  *buffNameX, unsigned int offX,
				IBuffer  *buffNameY, unsigned int offY,
				IBuffer  *buffNameZ, unsigned int offZ);

		protected:

			PassCompute(const std::string &passName);
			static bool Init();
			static bool Inited;

			std::shared_ptr<Material> m_Mat;
			IBuffer  *m_BufferX, *m_BufferY, *m_BufferZ;
			unsigned int m_OffsetX, m_OffsetY, m_OffsetZ;

		};
	};
};
#endif





	
	
