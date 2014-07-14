#ifndef IRENDERER_H
#define IRENDERER_H

#include <vector>
#include <string>

#include <nau/material/imaterialgroup.h>
#include <nau/material/material.h>
#include <nau/material/colormaterial.h>

#include <nau/math/vec4.h>
#include <nau/math/itransform.h>
#include <nau/geometry/frustum.h>
#include <nau/geometry/iboundingvolume.h>
#include <nau/scene/camera.h>
#include <nau/scene/iscene.h>
#include <nau/scene/sceneobject.h>
#include <nau/scene/light.h>
#include <nau/render/iprogram.h>
#include <nau/material/colormaterial.h>
#include <nau/enums.h>

using namespace nau::material;

namespace nau
{
	namespace render
	{
		class IRenderer
		{
		public:

			static const unsigned int MAX_COUNTERS = 32;
			static const unsigned int TRIANGLE_COUNTER = MAX_COUNTERS;

			enum TRenderMode {
				WIREFRAME_MODE = 0,
				POINT_MODE,
				SOLID_MODE,
				MATERIAL_MODE
			};

			typedef enum {
				PROJECTION_MATRIX,
				MODEL_MATRIX,
				VIEW_MATRIX,
				TEXTURE_MATRIX,
				COUNT_MATRIXMODE
			} MatrixMode;

			typedef enum {
				PROJECTION,
				MODEL,
				VIEW,
				TEXTURE,
				VIEW_MODEL,
				PROJECTION_VIEW_MODEL,
				PROJECTION_VIEW,
				TS05_PVM,
				NORMAL,
				COUNT_MATRIXTYPE
			} MatrixType;

			static const std::string MatrixTypeString[COUNT_MATRIXTYPE];

			static const std::string &getPropMatrixTypeString(MatrixType aType);

			static std::map<int, std::string> AtomicLabels;
			static int AtomicLabelsCount;
			static void addAtomic(int id, std::string name);


			enum TextureUnit {
				TEXTURE_UNIT0 = 0,
				TEXTURE_UNIT1,
				TEXTURE_UNIT2,
				TEXTURE_UNIT3,
				TEXTURE_UNIT4,
				TEXTURE_UNIT5,
				TEXTURE_UNIT6,
				TEXTURE_UNIT7,
				COUNT_TEXTUREUNIT
			} ;

			typedef enum {
				CLIP_PLANE0,
				CLIP_PLANE1,
				CLIP_PLANE2,
				CLIP_PLANE3
			} ClipPlane;

			typedef enum {
				FRONT,
				BACK,
				FRONT_AND_BACK
			} Face;


#if NAU_OPENGL_VERSION >= 400
			const static int PRIMITIVE_TYPE_COUNT = 8;
#else
			const static int PRIMITIVE_TYPE_COUNT = 7;
#endif

			enum DrawPrimitive{
				TRIANGLES=0,
				TRIANGLE_STRIP,
				TRIANGLE_FAN,
				LINES,
				LINE_LOOP,
				POINTS,
				TRIANGLES_ADJACENCY
#if NAU_OPENGL_VERSION >= 400
				, PATCH
#endif
			} ;

			typedef enum {
				COLOR_BUFFER = 0x01,
				DEPTH_BUFFER = 0x02,
				STENCIL_BUFFER = 0x04
			} Buffer;

			typedef enum {
				COLOR_CLEAR, 
				COLOR_ENABLE, 
				DEPTH_CLEAR, 
				DEPTH_ENABLE, 
				DEPTH_MASK, 
				DEPTH_CLAMPING,
				STENCIL_CLEAR, 
				STENCIL_ENABLE,

				COUNT_BOOL_PROPS} BoolProps;

			virtual void setProp(IRenderer::BoolProps prop, bool value) = 0;

			typedef enum {
				KEEP, 
				ZERO, 
				REPLACE, 
				INCR, 
				INCR_WRAP, 
				DECR, 
				DECR_WRAP, 
				INVERT} StencilOp;

			typedef enum {LESS, NEVER,ALWAYS,LEQUAL,
				EQUAL, GEQUAL, GREATER, NOT_EQUAL} StencilFunc;


			typedef enum {
				RENDER_MODE
			}Attribute;

			virtual float getDepthAtPoint(int x, int y) = 0;


			static void getPropId(std::string &s, int *id);
			//virtual void setCore(bool flag) {};

			virtual bool init() = 0;
			virtual void drawGroup (nau::material::IMaterialGroup* aMaterialGroup) = 0;
			virtual unsigned int translateDrawingPrimitive(unsigned int aDrawPrimitive) = 0;

			virtual void clear (unsigned int b) = 0;
			virtual void setDepthClearValue(float v) = 0;
			virtual void setDepthFunc(int f) = 0;
			//virtual void setDepthMask(bool b) = 0;
			virtual void setStencilClearValue(int v) = 0;
			virtual void setStencilMaskValue(int i) = 0;
			virtual void setStencilFunc(StencilFunc f, int ref, unsigned int mask) = 0;
			virtual void setStencilOp(StencilOp sfail, StencilOp dfail, StencilOp dpass) = 0;
			static int translateStringToStencilOp(std::string s);
			static int translateStringToStencilFunc(std::string s);
			virtual void setRenderMode (TRenderMode mode) = 0;


			virtual void setViewport(int width, int height) = 0;
			virtual void setViewport(nau::render::Viewport *aViewport) = 0;
			
			virtual void setCamera (nau::scene::Camera *aCamera) = 0;
			virtual Camera *getCamera() = 0;
			//virtual void loadIdentityAndSetCamera(nau::scene::Camera& aCamera) = 0;

			virtual void saveAttrib(Attribute aAttrib) = 0;
			virtual void restoreAttrib() = 0;

			/*Matrix Operations*/

			virtual void loadIdentity (void) = 0;
			virtual void setMatrixMode (MatrixMode mode) = 0;
			virtual const float *getMatrix(MatrixType aType) = 0;
			//virtual const float *getMatrix(MatrixDerived mode) = 0;
			//virtual const float *getNormalMatrix() = 0;

			virtual void pushMatrix (void) = 0;
			virtual void popMatrix (void) = 0;

			virtual void applyTransform (const nau::math::ITransform &aTransform) = 0;
			virtual void translate (nau::math::vec3 &aVec) = 0;
			virtual void scale (nau::math::vec3 &aVec) = 0;

			virtual float* getProjectionModelviewMatrix (void) = 0;

			//virtual void unproject (nau::render::IRenderable &aRenderable, nau::scene::Camera& aCamera) = 0;

			virtual void setCullFace (Face aFace) = 0;

			virtual void setColor (float r, float g, float b, float a) = 0;

			virtual void setColor (int r, int g, int b, int a) = 0;

			//virtual void setFixedFunction (bool fixed) = 0;


			virtual void flush (void) = 0;

			//virtual void enableStereo (void) = 0;
			//virtual void disableStereo (void) = 0;
			//virtual bool isStereo (void) = 0;

			/* Lights */



			virtual int getLightCount() = 0;
			virtual Light *getLight(unsigned int i) = 0;

			virtual bool addLight (nau::scene::Light& aLight) = 0;

			virtual void removeLights () = 0;


			/* Debug Operations */
			
			//virtual void renderBoundingVolume (const nau::geometry::IBoundingVolume* aBoundingVolume) = 0;

			virtual void resetCounters (void) = 0;

			virtual unsigned int getCounter (unsigned int c) = 0;


			// IMAGE TEXTURE
#if NAU_OPENGL_VERSION >=  420
			virtual void addImageTexture(unsigned int aTexUnit, ImageTexture *t) = 0;
			virtual void removeImageTexture(unsigned int aTexUnit) = 0;
			virtual int getImageTextureCount() = 0;
			virtual ImageTexture* getImageTexture(unsigned int unit) = 0;
#endif
			/* Textures */

			virtual void setActiveTextureUnit (TextureUnit aTexUnit) = 0;
			virtual void addTexture(TextureUnit aTexUnit, Texture *t) = 0;
			virtual void removeTexture(TextureUnit aTexUnit) = 0;
			virtual int getPropi(TextureUnit aTexUnit, Texture::IntProperty prop) = 0;
			virtual int getTextureCount() = 0;

#if NAU_CORE_OPENGL != 1
			virtual void enableTexturing (void) = 0; /***MARK***/ //Texture's dimension not specified
			virtual void disableTexturing (void) = 0;
			virtual void enableTextureCoordsGen (void) = 0;
			virtual void disableTextureCoordsGen (void) = 0;
#endif

			/* Framebuffer Operations */

			//virtual void enableDepthClamping (void) = 0;
			//virtual void disableDepthClamping (void) = 0;

			//virtual void enableDepthTest (void) = 0;
			//virtual void disableDepthTest (void) = 0;
			
			virtual void colorMask (bool r, bool g, bool b, bool a) = 0;

			virtual nau::math::vec3 readpixel (int x, int y) = 0;
	
			/* Frustum Operation */ 
			
			virtual void activateUserClipPlane (ClipPlane aClipPlane) = 0;

			virtual void setUserClipPlane (ClipPlane aClipPlane, double *plane) = 0;

			virtual void deactivateUserClipPlane (ClipPlane aClipPlane) = 0;

			/* Picking Operations */

			/* Material Operations */

			//! Set all color properties
			virtual void setMaterial(const nau::material::ColorMaterial &mat) = 0;
			virtual void setMaterial(float *diffuse, float *ambient, float *emission, float *specular, float shininess) = 0;
			virtual const float * getColor(ColorMaterial::ColorComponent aColor) = 0;
			//virtual void setMaterial (Face aFace, MaterialComponent aMaterialComponent, float *values) = 0;
			//
			//virtual void setMaterial (Face aFace, MaterialComponent aMaterialComponent, float value) = 0;

			virtual void setState (IState *aState) = 0; 
			virtual void setDefaultState() = 0;
			
			virtual ~IRenderer() {}

			/*  Shaders */
			virtual void setShader (IProgram *aShader) = 0; 
			virtual int getAttribLocation(std::string name) = 0;


#if NAU_CORE_OPENGL == 0
			virtual void activateLighting (void) = 0;
			virtual void deactivateLighting (void) = 0;
			virtual void positionLight (nau::scene::Light& aLight) = 0;

			virtual void enableFog (void) = 0; 
			virtual void disableFog (void) = 0;

#endif

			//virtual void disableSurfaceShaders (void) = 0;
			//virtual void enableSurfaceShaders (void) = 0;

			/* Draw Operations */

			//virtual void sort (nau::render::IRenderable &aRenderable) = 0;
			//virtual void startRender (void) = 0;
			//virtual void finishRender (void) = 0;
			//virtual void renderObject (nau::render::IRenderable &aRenderable, unsigned int buffers = 0) = 0;
			//virtual void renderObject (nau::scene::ISceneObject &aSceneObject, unsigned int buffers = 0) = 0;
			//virtual void renderMaterialGroup (nau::material::IMaterialGroup* aMaterialGroup) = 0;
			//virtual void finish (void) = 0;
			//virtual void drawElements (unsigned int size, std::vector<unsigned int>& indices) = 0;
			//virtual void setMaterialProvider (nau::scene::IScene *aScene) = 0; /***MARK***/ //!!!!!!!
			//virtual void activateDefaultLight (void) = 0;
			//virtual void deactivateDefaultLight (void) = 0;


		};
	};
};

#endif
