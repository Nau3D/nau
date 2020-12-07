#ifndef PASS_MESH_H
#define PASS_MESH_H

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

		class PassMesh : public Pass {
			friend class PassFactory;

		public:

			virtual ~PassMesh();

			static std::shared_ptr<Pass> Create(const std::string &name);

			void eventReceived(const std::string &sender, const std::string &eventType, 
				const std::shared_ptr<IEventData> &evt);

			const std::string &getClassName();

			void prepare();
			void restore();
			void doPass();

			void setMaterialName(const std::string &lName,const std::string &mName);
			std::shared_ptr<Material> &getMaterial();

			void setDimension(unsigned int dimX);
			void setDimFromBuffer(IBuffer  *buffNameX, unsigned int offX);

		protected:

			PassMesh(const std::string &passName);
			static bool Init();
			static bool Inited;

			std::shared_ptr<Material> m_Mat;
			IBuffer  *m_Buffer;
			unsigned int m_Offset;

		};
	};
};
#endif





	
	
