#ifndef LISPSMALGORITHM
#define LISPSMALGORITHM

#include "nau/render/irenderalgorithm.h"
#include "nau/scene/iscene.h"
#include "nau/render/irenderer.h"
#include "nau/render/texture.h"
#include "nau/render/rendertarget.h"
#include "nau/geometry/quad.h"

#include "nau/math/simpletransform.h"

namespace nau
{
	namespace render
	{
		class LiSPSMAlgorithm :
			public IRenderAlgorithm
		{
		public:
			LiSPSMAlgorithm(void);

			void init (void);
			void renderScene (nau::scene::IScene *aScene);
			void setRenderer (nau::render::IRenderer *aRenderer);

			void externCommand (char keyCode);
		public:
			~LiSPSMAlgorithm(void);
		
		private:
			void calculateShadow (nau::scene::IScene *aScene, nau::scene::Camera &aCamera);
			void materialPass (std::vector<nau::scene::ISceneObject*> &sceneObjects, nau::math::ITransform  &t);
			void deferredShadePass (nau::scene::Camera &quadCam, nau::scene::Camera &aCamera, nau::scene::Light &aLight);
			void drawShadow (nau::scene::Camera &quadCam, nau::scene::Camera &lightCam, nau::scene::Camera &aCamera);

			void debug (nau::scene::Camera &quadCam);
			void waterPass (std::vector<nau::scene::ISceneObject*> &sceneObjects);

			bool waterOnFrustum (nau::scene::IScene *aScene, std::vector<nau::scene::ISceneObject*> &sceneObjects, float *plane);
			void renderFixed (std::vector<nau::scene::ISceneObject*> &sceneObjects, nau::scene::Camera &aCam);

		private:
			bool m_Inited;

			nau::render::IRenderer *m_pRenderer;
			nau::render::Texture *m_ShadowTexture;
			nau::render::Texture *m_DepthTexture;
			nau::render::Texture *m_AmbientTexture;
			nau::render::Texture *m_NormalTexture;
			nau::render::Texture *m_PositionTexture;
			nau::render::Texture *m_WaterReflectionTexture;
			nau::render::Texture *m_WaterDepthTexture;
			nau::render::Texture *m_FinalTexture;
			nau::render::Texture *m_LightCamTexture;
			nau::render::RenderTarget *m_WaterFBO;
			nau::render::RenderTarget *m_RenderTarget;
			nau::render::RenderTarget *m_MRT;

			CProgram *m_pBlankShader;
			CProgram *m_pDeferredShader;
			CProgram *m_pDeferredShadowShader;
			CProgram *m_pMaterialPassShader;
			CProgram *m_pWaterShader;

			CTexture *m_CausticTexture;

			CMaterial *m_WaterMaterial; /***MARK***/ /*HACK!!!!*/

			nau::geometry::Quad *m_Quad;


			nau::math::SimpleTransform m_LightTransforms[4];
			int m_Split;
			bool m_RenderBoundBox;
			bool m_FixedFunc;
		};
	};
};

#endif //LISPSMALGORITHM
