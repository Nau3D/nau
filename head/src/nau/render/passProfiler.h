#ifndef PROFILERPASS_H
#define PROFILERPASS_H

#include "nau/render/pass.h"
#include "nau/resource/font.h"
#include "nau/scene/camera.h"
#include "nau/render/irenderable.h"

namespace nau
{
	namespace render
	{
		class PassProfiler :
			public Pass
		{
		private:
			nau::resource::Font m_pFont;
			nau::scene::Camera *m_pCam;
			nau::scene::SceneObject *m_pSO;
		public:
			PassProfiler (const std::string &name);

			void prepare (void);
			void restore (void);
			void doPass (void);
			virtual void setCamera (const std::string &cameraName);
			
			virtual ~PassProfiler(void);

		};
	};
};
#endif 
