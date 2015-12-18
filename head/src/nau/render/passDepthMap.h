#ifndef DEPTHMAPPASS_H
#define DEPTHMAPPASS_H



#include "nau/render/pass.h"

namespace nau
{
	namespace render
	{
		class PassDepthMap : public Pass {

			friend class PassFactory;

		protected:
			std::shared_ptr<nau::scene::Camera> m_LightCamera;

			static bool Init();
			static bool Inited;
			PassDepthMap(const std::string &name);

		public:

			~PassDepthMap(void);

			static std::shared_ptr<Pass> Create(const std::string &name);

			virtual void prepare (void);
			virtual void doPass (void);
			virtual void restore (void);
			virtual void addLight (const std::string &light);
		};
	};
};
#endif //DEPTHMAPPASS_H
