#ifndef GLRENDERER_H
#define GLRENDERER_H

#include <GL/glew.h>
//#include <GL/gl.h>
//#include <GL/glu.h>

#include "nau/render/irenderer.h"
#include "nau/math/simpletransform.h"
#include "nau/math/matrix.h"

#include "nau/geometry/frustum.h"
#include "nau/scene/camera.h"
#include "nau/render/iprogram.h"
#include "nau/material/material.h"
#include "nau/material/colormaterial.h"
#include "nau/math/itransform.h"
#include "nau/config.h"
#include "nau/render/imageTexture.h"
#include "nau/render/pass.h"


#define LOGGING_ON
#include "nau/clogger.h"
//#undef LOGGING_ON

#include "nau/render/opengl/glstate.h"


using namespace nau::scene;
using namespace nau::render;

namespace nau
{
	namespace render
	{

		class GLRenderer : public nau::render::IRenderer
		{
		public:
			GLRenderer();
			~GLRenderer(void);

#if (NAU_OPENGL_VERSION >= 400)
		// ATOMIC COUNTERS
		protected:
			//GLuint m_AtomicCountersBuffer;
			//void prepareAtomicCounterBuffer();
			//void resetAtomicCounters();
			//void readAtomicCounters();
			/// Array to store atomic counters
			std::vector<unsigned int> m_AtomicCounterValues;

		public:
			std::vector<unsigned int> &getAtomicCounterValues();
#endif


		protected:
			static bool Init();
			static bool Inited;

		public:
			static unsigned int GLPrimitiveTypes[PRIMITIVE_TYPE_COUNT];
			//void setPassPropb(Pass::BoolProps prop, bool value);
			//bool getPassPropb(Pass::BoolProps prop);

			//! \name Methods
			//@{
			bool init();

			virtual int getNumberOfPrimitives(MaterialGroup *m) ;

			// RENDER
			void setRenderMode (TRenderMode mode);
			void drawGroup (nau::material::MaterialGroup* aMatGroup);
			void clearFrameBuffer(unsigned int b);
			void prepareBuffers(Pass *p);
			void setDepthClamping(bool b);

			//void setDepthClearValue(float v);
			//void setDepthFunc(int f);
			//void setStencilClearValue(int v);
			//void setStencilMaskValue(int i);
			//void setStencilFunc(StencilFunc f, int ref, unsigned int mask);
			//void setStencilOp(StencilOp sfail, StencilOp dfail, StencilOp dpass);

			// PRIMITIVE COUNTER
			void resetCounters (void);
			unsigned int getCounter(Counters c);

			// RENDER ATTRIBS
			void saveAttrib(RendererAttributes aAttrib);
			void restoreAttrib();
			virtual void setCullFace (Face aFace);
			void colorMask (bool r, bool g, bool b, bool a);

			// SHADERS
			void setShader (IProgram *aShader); 
			int getAttribLocation(std::string name);

			// CAMERA
			void setCamera (nau::scene::Camera *aCamera);
			Camera *getCamera();
			nau::geometry::Frustum& getFrustum (void);
			//void setViewport(int width, int height);
			void setViewport(nau::render::Viewport *vp);	
			Viewport *getViewport();

			// MATRICES
			void loadIdentity(MatrixMode mode);
			const float *getMatrix(MatrixType aType);

			void pushMatrix(MatrixMode mode);
			void popMatrix(MatrixMode mode);

			void applyTransform(MatrixMode mode, const nau::math::ITransform &aTransform) ;
			void translate(MatrixMode mode, nau::math::vec3 &aVec) ;
			void scale(MatrixMode mode, nau::math::vec3 &aVec) ;
			void rotate(MatrixMode mode, float angle, nau::math::vec3 &axis) ;


			virtual float getDepthAtPoint(int x, int y);

			// IMAGE TEXTURE
#if NAU_OPENGL_VERSION >=  420
			void addImageTexture(unsigned int aTexUnit, ImageTexture *t);
			void removeImageTexture(unsigned int aTexUnit);
			int getImageTextureCount();
			ImageTexture* getImageTexture(unsigned int unit);
#endif
			// TEXTURING
			void addTexture(unsigned int aTexUnit, Texture *t);
			void removeTexture(unsigned int aTexUnit);
			int getPropi(unsigned int aTexUnit, Texture::IntProperty prop);
			int getTextureCount();
			void setActiveTextureUnit(unsigned int aTexUnit);
			Texture *getTexture(int unit);

			// STATE
			void setState (IState *aState);
			void setDefaultState();
			IState *getState();

			// LIGHTING
			virtual bool addLight (nau::scene::Light& aLight);
			virtual void removeLights ();
			virtual int getLightCount();
			virtual Light *getLight(unsigned int id);

			// COLOR AND MATERIALS
			const vec4 &getColorProp4f(nau::material::ColorMaterial::Float4Property prop);
			float getColorPropf(nau::material::ColorMaterial::FloatProperty prop);
			float *getColorProp(int prop, Enums::DataType dt);
			virtual void setMaterial( nau::material::ColorMaterial &mat);
			virtual void setMaterial(float *diffuse, float *ambient, float *emission, float *specular, float shininess);
			//virtual const float * getColor(ColorMaterial::ColorComponent aColor);
			virtual void setColor (float r, float g, float b, float a);
			virtual void setColor (int r, int g, int b, int a);


			//CLIP PLANES
			void activateUserClipPlane(unsigned int  aClipPlane);
			void setUserClipPlane(unsigned int  aClipPlane, double *plane);
			void deactivateUserClipPlane(unsigned int  aClipPlane);

			nau::math::vec3 readpixel (int x, int y);

			void flush (void);

			unsigned int translateStencilDepthFunc(int aFunc);
			unsigned int translateStencilOp(int aFunc);

		private:


			Viewport *m_Viewport;

			std::vector<Light *> m_Lights;
			Camera *m_Camera;

			unsigned int m_LightsOn;

			std::vector<Texture *> m_Textures;
#if NAU_OPENGL_VERSION >=  420
			std::vector<ImageTexture *> m_ImageTextures;
#endif
			std::vector<SimpleTransform> m_Matrices;
			ITransform *m_CurrentMatrix;

			// pre alocated memory to return composed matrices
			mat4 m_pReturnMatrix;
			mat3 m_pReturnMat3;
			float m_fDummy;
			vec4 m_vDummy;

			std::vector<SimpleTransform> m_MatrixStack[IRenderer::COUNT_MATRIXMODE];

			int m_TriCounter;
			unsigned int *userCounters;
			void accumTriCounter(unsigned int drawPrimitive, unsigned int size);

			nau::render::GlState m_glCurrState, m_glDefaultState; 
			float m_ReturnMatrix[16];

			nau::material::ColorMaterial m_Material;

			int m_TexturingFlag;
			IRenderer::TRenderMode m_PrevRenderMode, m_ActualRenderMode;

			IProgram *m_Shader; 


			GLenum translateFace (Face aFace);
			unsigned int translateDrawingPrimitive(unsigned int aDrawPrimitive);

		};
	};
};

#endif
