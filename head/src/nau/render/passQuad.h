#ifndef QUADPASS_H
#define QUADPASS_H

#include "nau/render/pass.h"
#include "nau/geometry/quad.h"

namespace nau {

	namespace render {

		class PassQuad : public Pass {
			friend class PassFactory;
		public:
			~PassQuad(void);

			static std::shared_ptr<Pass> Create(const std::string &name);

			void prepare (void);
			void restore (void);
			void doPass (void);

			void setMaterialName(std::string &lib, std::string &mat);
			

		protected:
			PassQuad (const std::string &name);
			nau::geometry::Quad *m_QuadObject;
			static bool Init();
			static bool Inited;

		};
	};
};
#endif 
