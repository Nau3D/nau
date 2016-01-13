#ifndef PROFILERPASS_H
#define PROFILERPASS_H

#include "nau/geometry/font.h"
#include "nau/render/pass.h"
#include "nau/render/iRenderable.h"
#include "nau/scene/camera.h"

namespace nau
{
	namespace render
	{
		class PassProfiler : public Pass {
			friend class PassFactory;

		protected:
			PassProfiler (const std::string &name);
			nau::geometry::Font m_pFont;
			std::shared_ptr<nau::scene::Camera> m_pCam;
			std::shared_ptr<SceneObject> m_pSO;

			static bool Init();
			static bool Inited;

		public:
			virtual ~PassProfiler(void);

			static std::shared_ptr<Pass> Create(const std::string &name);

			void prepare (void);
			void restore (void);
			void doPass (void);
			virtual void setCamera (const std::string &cameraName);
			

		};
	};
};
#endif 
