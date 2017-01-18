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

			void eventReceived(const std::string & sender, const std::string & eventType, 
				const std::shared_ptr<IEventData>& evt);

			void prepare (void);
			void restore (void);
			void doPass (void);

			void setMaterialName(std::string &lib, std::string &mat);
			

		protected:
			PassQuad (const std::string &name);
			std::shared_ptr<nau::scene::SceneObject> m_QuadObject;
			//nau::geometry::Quad *m_QuadObject;
			static bool Init();
			static bool Inited;

		};
	};
};
#endif 
