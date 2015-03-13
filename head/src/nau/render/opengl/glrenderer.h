#ifndef GLRENDERER_H
#define GLRENDERER_H

#include <GL/glew.h>

#include "nau/config.h"
#include "nau/geometry/frustum.h"
#include "nau/material/material.h"
#include "nau/material/colormaterial.h"
#include "nau/math/matrix.h"
#include "nau/render/imageTexture.h"
#include "nau/render/iprogram.h"
#include "nau/render/irenderer.h"
#include "nau/render/pass.h"
#include "nau/render/opengl/glstate.h"
#include "nau/scene/camera.h"


#define LOGGING_ON
#include "nau/clogger.h"
//#undef LOGGING_ON



using namespace nau::scene;
using namespace nau::render;

namespace nau
{
	namespace render
	{

		class GLRenderer : public nau::render::IRenderer
		{
		protected:
			static bool Init();
			static bool Inited;

		public:
			static unsigned int GLPrimitiveTypes[PRIMITIVE_TYPE_COUNT];

			GLRenderer();
			~GLRenderer(void);

			bool init();

			void *getProp(unsigned int prop, Enums::DataType dt);
			const mat4 &getPropm4(Mat4Property prop);
			const mat3 &getPropm3(Mat3Property prop);

			// ATOMIC COUNTERS

#if (NAU_OPENGL_VERSION >= 400)
		protected:
			/// Array to store atomic counters
			std::vector<unsigned int> m_AtomicCounterValues;

		public:
			std::vector<unsigned int> &getAtomicCounterValues();
#endif

			// LIGHTS

		protected:
			std::vector<Light *> m_Lights;

		public:
			virtual int getLightCount();
			virtual Light *getLight(unsigned int id);
			virtual bool addLight(nau::scene::Light& aLight);
			virtual void removeLights();


			// CAMERA

		protected:
			Viewport *m_Viewport;
			Camera *m_Camera;

		public:
			void setViewport(nau::render::Viewport *vp);
			Viewport *getViewport();
			void setCamera(nau::scene::Camera *aCamera);
			Camera *getCamera();


			// COUNTERS

		protected:
			int m_TriCounter;
			unsigned int *userCounters;
			void accumTriCounter(unsigned int drawPrimitive, unsigned int size);
		public:
			void resetCounters(void);
			unsigned int getCounter(Counters c);
			virtual int getNumberOfPrimitives(MaterialGroup *m);


			// MATRICES

		protected:
			std::vector<mat4> m_MatrixStack[IRenderer::COUNT_MATRIXMODE];
			// pre alocated memory to return composed matrices
			mat4 m_pReturnMatrix;
			mat3 m_pReturnMat3;
		public:
			void loadIdentity(MatrixMode mode);
			void pushMatrix(MatrixMode mode);
			void popMatrix(MatrixMode mode);
			void applyTransform(MatrixMode mode, const nau::math::mat4 &aTransform);
			void translate(MatrixMode mode, nau::math::vec3 &aVec);
			void scale(MatrixMode mode, nau::math::vec3 &aVec);
			void rotate(MatrixMode mode, float angle, nau::math::vec3 &axis);


			// MATERIAL (color, state, shaders and textures)

		protected:
			std::map<int, MaterialTexture *> m_Textures;
#if NAU_OPENGL_VERSION >=  420
			std::map<int, ImageTexture *> m_ImageTextures;
#endif
			nau::render::GlState m_glCurrState, m_glDefaultState;
			nau::material::ColorMaterial m_Material;
			IProgram *m_Shader;

		public:
			// COLOR
			virtual void setMaterial(nau::material::ColorMaterial &mat);
			virtual void setMaterial(float *diffuse, float *ambient, float *emission, float *specular, float shininess);
			virtual ColorMaterial *getMaterial();

			// STATE
			void setState(IState *aState);
			void setDefaultState();
			IState *getState();

			// SHADER
			void setShader(IProgram *aShader);
			int getAttribLocation(std::string name);

			// IMAGE TEXTURE
#if NAU_OPENGL_VERSION >=  420
			void addImageTexture(unsigned int aTexUnit, ImageTexture *t);
			void removeImageTexture(unsigned int aTexUnit);
			int getImageTextureCount();
			ImageTexture* getImageTexture(unsigned int unit);
#endif
			// TEXTURING
			void setActiveTextureUnit(unsigned int aTexUnit);
			void addTexture(MaterialTexture *t);
			void removeTexture(unsigned int aTexUnit);
			MaterialTexture *getMaterialTexture(int unit);
			Texture *getTexture(int unit);
			int getTextureCount();


			// FRAMEBUFFER OPS

			void clearFrameBuffer(unsigned int b);
			void prepareBuffers(Pass *p);
			void flush(void);

			void setDepthClamping(bool b);
			void colorMask (bool r, bool g, bool b, bool a);


			// RENDER

		protected:
			int m_TexturingFlag;
			IRenderer::TRenderMode m_PrevRenderMode, m_ActualRenderMode;

		public:
			void setRenderMode(TRenderMode mode);
			void drawGroup (nau::material::MaterialGroup* aMatGroup);
			virtual void setCullFace (Face aFace);


			// RENDER ATTRIBS

			void saveAttrib(RendererAttributes aAttrib);
			void restoreAttrib();


			// MISC

			virtual float getDepthAtPoint(int x, int y);
			nau::math::vec3 readpixel (int x, int y);


			unsigned int translateStencilDepthFunc(int aFunc);
			unsigned int translateStencilOp(int aFunc);

		protected:

			GLenum translateFace (Face aFace);
			unsigned int translateDrawingPrimitive(unsigned int aDrawPrimitive);

		};
	};
};

#endif
