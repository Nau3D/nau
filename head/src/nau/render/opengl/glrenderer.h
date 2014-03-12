#ifndef GLRENDERER_H
#define GLRENDERER_H

#include <GL/glew.h>
//#include <GL/gl.h>
//#include <GL/glu.h>

#include <nau/render/irenderer.h>
#include <nau/math/simpletransform.h>
#include <nau/math/mat4.h>
#include <nau/math/mat3.h>

#include <nau/geometry/frustum.h>
#include <nau/scene/camera.h>
#include <nau/render/iprogram.h>
#include <nau/material/material.h>
#include <nau/material/colormaterial.h>
#include <nau/math/itransform.h>
#include <nau/config.h>
#include <nau/render/imageTexture.h>


#define LOGGING_ON
#include <nau/clogger.h>
//#undef LOGGING_ON

#include <nau/render/opengl/glstate.h>

using namespace nau::scene;

namespace nau
{
	namespace render
	{

		class GLRenderer : public nau::render::IRenderer
		{
		public:
			GLRenderer();
			~GLRenderer(void);

			static unsigned int GLPrimitiveTypes[PRIMITIVE_TYPE_COUNT];

			void setProp(IRenderer::BoolProps prop, bool value);
			bool getPropb(IRenderer::BoolProps prop);


			//! \name Methods
			//@{
			bool init();


			// RENDER
			void setRenderMode (TRenderMode mode);
			void drawGroup (nau::material::IMaterialGroup* aMatGroup);
			void clear (unsigned int b);

			void setDepthClearValue(float v);
			void setDepthFunc(int f);

			void setStencilClearValue(int v);
			void setStencilMaskValue(int i);
			void setStencilFunc(StencilFunc f, int ref, unsigned int mask);
			void setStencilOp(StencilOp sfail, StencilOp dfail, StencilOp dpass);

			// PRIMITIVE COUNTER
			void resetCounters (void);
			unsigned int getCounter (unsigned int c);

			// RENDER ATTRIBS
			void saveAttrib(Attribute aAttrib);
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
			void setViewport(int width, int height);
			void setViewport(nau::render::Viewport *vp);	

			// MATRICES
			void setMatrixMode (MatrixMode mode);
			void loadIdentity (void);
			const float *getMatrix(MatrixType mode);

			void translate (nau::math::vec3 &aVec);
			void scale (nau::math::vec3 &aVec);
			void rotate(float angle, nau::math::vec3 &axis);
			void applyTransform (const nau::math::ITransform &aTransform);

			void pushMatrix (void);
			void popMatrix (void);

			float* getProjectionModelviewMatrix (void);

			virtual float getDepthAtPoint(int x, int y);

			// IMAGE TEXTURE
#if NAU_OPENGL_VERSION >=  420
			void addImageTexture(unsigned int aTexUnit, ImageTexture *t);
			void removeImageTexture(unsigned int aTexUnit);
			int getImageTextureCount();
			ImageTexture* getImageTexture(unsigned int unit);
#endif
			// TEXTURING
			void addTexture(TextureUnit aTexUnit, Texture *t);
			void removeTexture(TextureUnit aTexUnit);
			int getPropi(TextureUnit aTexUnit, Texture::IntProperty prop);
			int getTextureCount();
			void setActiveTextureUnit (TextureUnit aTexUnit);

#if NAU_CORE_OPENGL != 1
			void enableTexturing (void);
			void disableTexturing (void);
			void enableTextureCoordsGen (void);
			void disableTextureCoordsGen (void);
#endif
			// STATE
			void setState (IState *aState);
			void setDefaultState();

			// LIGHTING
			virtual bool addLight (nau::scene::Light& aLight);
			virtual void removeLights ();
			virtual int getLightCount();
			virtual Light *getLight(unsigned int id);

			// COLOR AND MATERIALS
			virtual void setMaterial(const nau::material::ColorMaterial &mat);
			virtual void setMaterial(float *diffuse, float *ambient, float *emission, float *specular, float shininess);
			virtual const float * getColor(ColorMaterial::ColorComponent aColor);
			virtual void setColor (float r, float g, float b, float a);
			virtual void setColor (int r, int g, int b, int a);


			//CLIP PLANES
			void activateUserClipPlane (ClipPlane aClipPlane);
			void setUserClipPlane (ClipPlane aClipPlane, double *plane);
			void deactivateUserClipPlane (ClipPlane aClipPlane);

			nau::math::vec3 readpixel (int x, int y);

			void flush (void);

#if NAU_CORE_OPENGL == 0
			// FOG
			virtual void enableFog (void); 
			virtual void disableFog (void);

			//LIGHTING
			virtual void activateLighting (void);
			virtual void deactivateLighting (void);
			virtual void positionLight (nau::scene::Light& aLight);
#endif
			//void setDepthMask(bool b);
			//void enableDepthTest (void);
			//void disableDepthTest (void);
			//void enableDepthClamping (void);
			//void disableDepthClamping (void);
			//virtual void setCore(bool flag);
			//static void getPropTypeAndId(std::string &s, MatrixType *dt , int *id);
			//virtual void setFixedFunction (bool fixed);
			// STEREO
			//void enableStereo (void);
			//void disableStereo (void);
			//bool isStereo (void);

			// MISC
			//void renderBoundingVolume (const nau::geometry::IBoundingVolume* aBoundingVolume);

		private:

			unsigned int translate(StencilFunc aFunc);
			unsigned int translate(StencilOp aFunc);

			std::vector<Light *> m_Lights;
			Camera *m_Camera;

			unsigned int m_LightsOn;

			std::vector<Texture *> m_Textures;
#if NAU_OPENGL_VERSION >=  420
			std::vector<ImageTexture *> m_ImageTextures;
#endif
			std::vector<SimpleTransform> m_Matrices;
			IRenderer::MatrixMode m_MatrixMode;
			ITransform *m_CurrentMatrix;

			// pre alocated memory to return composed matrices
			mat4 m_pReturnMatrix;
			mat3 m_pReturnMat3;

			std::vector<SimpleTransform> m_MatrixStack[IRenderer::COUNT_MATRIXMODE];

			GLuint m_AtomicCountersBuffer;
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
			GLenum translateMaterialComponent (ColorMaterial::ColorComponent aMaterialComponent);
			unsigned int translateDrawingPrimitive(unsigned int aDrawPrimitive);

			//bool m_FixedFunction;
			//bool m_Stereo;
			//void doRender (nau::scene::ISceneObject *aRenderable, unsigned int buffers, int priority);
			//std::multimap<int, nau::render::IRenderable*> m_RenderQueue;
			//void setDepthCompare();
			//std::map<int, std::map<Material*, std::vector<std::pair<nau::materials::IMaterialGroup*, nau::math::ITransform*>>*>* > m_RenderQueue;
			//void drawElements (unsigned int size, std::vector<unsigned int>& indices, unsigned int aDrawingPrimitive = TRIANGLES);
			//virtual void setMaterialProvider (nau::scene::IScene *aScene); /***MARK***/ //!!!!!!!
			//virtual void activateDefaultLight (void);
			//virtual void deactivateDefaultLight (void);
		};
	};
};

#endif
