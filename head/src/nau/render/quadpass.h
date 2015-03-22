#ifndef QUADPASS_H
#define QUADPASS_H

#include "nau/render/pass.h"
#include "nau/geometry/quad.h"

namespace nau
{
	namespace render
	{
		class QuadPass :
			public Pass
		{
		private:
			nau::geometry::Quad *m_QuadObject;
		public:
			QuadPass (const std::string &name);

			void prepare (void);
			void restore (void);
			void doPass (void);

			void setMaterialName(std::string &lib, std::string &mat);
			
			~QuadPass(void);

		};
	};
};
#endif 
