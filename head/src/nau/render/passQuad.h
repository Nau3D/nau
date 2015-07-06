#ifndef QUADPASS_H
#define QUADPASS_H

#include "nau/render/pass.h"
#include "nau/geometry/quad.h"

namespace nau
{
	namespace render
	{
		class PassQuad :
			public Pass
		{
		private:
			nau::geometry::Quad *m_QuadObject;
		public:
			PassQuad (const std::string &name);

			void prepare (void);
			void restore (void);
			void doPass (void);

			void setMaterialName(std::string &lib, std::string &mat);
			
			~PassQuad(void);

		};
	};
};
#endif 
