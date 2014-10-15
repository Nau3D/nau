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


#include <nau/attribute.h>
#include <nau/attributeValues.h>

using namespace nau::material;
using namespace nau::render;

namespace nau
{
	namespace render
	{
		class Pass;
		class IRenderer: public AttributeValues
		{

		protected:
			static bool Init();
			static bool Inited;

		public:	

			static AttribSet MatrixAttribs;

			ENUM_PROP(STENCIL_FUNC, 0);
			ENUM_PROP(STENCIL_FAIL, 1);
			ENUM_PROP(STENCIL_DEPTH_FAIL, 2);
			ENUM_PROP(STENCIL_DEPTH_PASS, 3);
			ENUM_PROP(DEPTH_FUNC, 4);


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

			static int MaxTextureUnits;
			static int MaxColorAttachments;


			typedef enum {
				FRONT,
				BACK,
				//FRONT_AND_BACK
			} Face;


#if NAU_OPENGL_VERSION >= 400
			const static int PRIMITIVE_TYPE_COUNT = 8;
#else
			const static int PRIMITIVE_TYPE_COUNT = 7;
#endif


			typedef enum {
				COLOR_BUFFER = 0x01,
				DEPTH_BUFFER = 0x02,
				STENCIL_BUFFER = 0x04
			} FrameBuffer;

			//typedef enum {
			//	COLOR_CLEAR, 
			//	COLOR_ENABLE, 
			//	DEPTH_CLEAR, 
			//	DEPTH_ENABLE, 
			//	DEPTH_MASK, 
			//	DEPTH_CLAMPING,
			//	STENCIL_CLEAR, 
			//	STENCIL_ENABLE,

			//	COUNT_BOOL_PROPS} BoolProps;

			//virtual void setPassProp(Pass::BoolProperty prop, bool value) = 0;


			void setPrope(EnumProperty prop, int value);
			void setProp(int prop, Enums::DataType type, void *value);
			


		// ATOMIC COUNTERS 

#if NAU_OPENGL_VERSION >= 400
		public:
			/// Number of Atomic Counters and
			unsigned int m_AtomicCount = 0;
			///  Max Atomic Counter ID	
			unsigned int m_AtomicMaxID = 0;
			/// name of atomic counters
			std::map<int, std::string> m_AtomicLabels;

			/// add an atomic counter
			void addAtomic(unsigned int id, std::string name);
			/// returns the atomic id, or -1 if the name is not defined
			int getAtomicID(std::string);
			/// get atomic counter values
			virtual unsigned int *getAtomicCounterValues() = 0;

		protected:
			/// Array to store atomic counters
			unsigned int *m_AtomicCounterValues = NULL;
			/// flag indicating if Atomic Buffer is created
			bool m_AtomicBufferPrepared = false;
#endif

		// LIGHTS

		public:
			/// returns the number of active lights
			virtual int getLightCount() = 0;
			/// returns light index i
			virtual Light *getLight(unsigned int i) = 0;
			/// adds a light at the next free index
			virtual bool addLight(nau::scene::Light& aLight) = 0;
			/// removes all lights
			virtual void removeLights() = 0;


		// CAMERAS

		public:
			/// sets viewport (origin is 0,0)
			//virtual void setViewport(int width, int height) = 0;
			/// sets viewport
			virtual void setViewport(nau::render::Viewport *aViewport) = 0;
			/// returns currrent viewport
			virtual Viewport *getViewport() = 0;
			/// set the camera
			virtual void setCamera(nau::scene::Camera *aCamera) = 0;
			/// returns current camera
			virtual Camera *getCamera() = 0;


		// COUNTERS

		public:
			typedef enum {
				TRIANGLE_COUNTER
			} Counters;

			/// resets all counters
			virtual void resetCounters(void) = 0;
			/// returns counter value
			virtual unsigned int getCounter(Counters c) = 0;


		// MATRICES

		public:
			typedef enum {
				PROJECTION_MATRIX,
				MODEL_MATRIX,
				VIEW_MATRIX,
				TEXTURE_MATRIX,
				COUNT_MATRIXMODE
			} MatrixMode;

			/// loads the identity matrix into matrix "mode"
			virtual void loadIdentity(MatrixMode mode) = 0;
			/// returns a float array with matrix values
			virtual const float *getMatrix(MatrixType aType) = 0;
			/// pushes matrix into matrix stack
			virtual void pushMatrix(MatrixMode mode) = 0;
			/// pops matrix into matrix stack
			virtual void popMatrix(MatrixMode mode) = 0;
			/// compose Current = Current * aTransform
			virtual void applyTransform(MatrixMode mode, const nau::math::ITransform &aTransform) = 0;
			/// compose Current = Current * translate(vec)
			virtual void translate(MatrixMode mode, nau::math::vec3 &aVec) = 0;
			/// compose Current = Current * scale(vec)
			virtual void scale(MatrixMode mode, nau::math::vec3 &aVec) = 0;
			/// compose Current = Current * rotate(angle,axis)
			virtual void rotate(MatrixMode mode, float angle, nau::math::vec3 &axis) = 0;


		// MATERIAL (color, state, shaders and textures)

		public:
			// color
			virtual void setColor(float r, float g, float b, float a) = 0;
			virtual void setColor(int r, int g, int b, int a) = 0;
			virtual void setMaterial(nau::material::ColorMaterial &mat) = 0;
			virtual void setMaterial(float *diffuse, float *ambient, float *emission, float *specular, float shininess) = 0;
			virtual const vec4 &getColorProp4f(ColorMaterial::Float4Property) = 0;
			virtual float getColorPropf(ColorMaterial::FloatProperty) = 0;
			virtual float *getColorProp(int prop, Enums::DataType dt) = 0;

			// state
			virtual void setState(IState *aState) = 0;
			virtual void setDefaultState() = 0;
			virtual IState *getState() = 0;

			/// shaders
			virtual void setShader(IProgram *aShader) = 0;
			/// returns the attribute location of a uniform var
			virtual int getAttribLocation(std::string name) = 0;

			// image textures
#if NAU_OPENGL_VERSION >=  420
			virtual void addImageTexture(unsigned int aTexUnit, ImageTexture *t) = 0;
			virtual void removeImageTexture(unsigned int aTexUnit) = 0;
			virtual int getImageTextureCount() = 0;
			virtual ImageTexture* getImageTexture(unsigned int unit) = 0;
#endif

			// textures 
			virtual void setActiveTextureUnit(unsigned int aTexUnit) = 0;
			virtual void addTexture(unsigned int aTexUnit, Texture *t) = 0;
			virtual void removeTexture(unsigned int aTexUnit) = 0;
			virtual int getPropi(unsigned int aTexUnit, Texture::IntProperty prop) = 0;
			virtual int getTextureCount() = 0;
			virtual Texture *getTexture(int unit) = 0;
			
		
		// FRAMEBUFFER OPS

		public:

			//typedef enum {
			//	KEEP,
			//	ZERO,
			//	REPLACE,
			//	INCR,
			//	INCR_WRAP,
			//	DECR,
			//	DECR_WRAP,
			//	INVERT
			//} StencilOp;

			//typedef enum {
			//	LESS, NEVER, ALWAYS, LEQUAL,
			//	EQUAL, GEQUAL, GREATER, NOT_EQUAL
			//} StencilFunc;

			virtual void clearFrameBuffer (unsigned int b) = 0;
			virtual void prepareBuffers(Pass *p) = 0;

			virtual unsigned int translateStencilDepthFunc(int aFunc) = 0;
			virtual unsigned int translateStencilOp(int aFunc) = 0;

			virtual void setDepthClamping(bool b) = 0;
			//virtual void setDepthClearValue(float v) = 0;
			//virtual void setDepthFunc(int f) = 0;

			//virtual void setStencilClearValue(int v) = 0;
			//virtual void setStencilMaskValue(int i) = 0;
			//virtual void setStencilFunc(StencilFunc f, int ref, unsigned int mask) = 0;
			//virtual void setStencilOp(StencilOp sfail, StencilOp dfail, StencilOp dpass) = 0;

			//static int translateStringToStencilOp(std::string s);
			//static int translateStringToStencilFunc(std::string s);

			virtual void colorMask(bool r, bool g, bool b, bool a) = 0;


		// RENDER

		public:
			typedef enum {
				RENDER_MODE
			}RendererAttributes;

			typedef enum  {
				WIREFRAME_MODE = 0,
				POINT_MODE,
				SOLID_MODE,
				MATERIAL_MODE
			} TRenderMode;


			virtual void setRenderMode(TRenderMode mode) = 0;
			virtual void drawGroup(nau::material::IMaterialGroup* aMaterialGroup) = 0;
			virtual void setCullFace(Face aFace) = 0;

			virtual void saveAttrib(RendererAttributes aAttrib) = 0;
			virtual void restoreAttrib() = 0;

		public:
			/// returns the number of primitives foa  material group
			virtual int getNumberOfPrimitives(IMaterialGroup *m) = 0;

			virtual float getDepthAtPoint(int x, int y) = 0;


			static void getPropId(std::string &s, int *id);
			//virtual void setCore(bool flag) {};

			virtual bool init() = 0;
			virtual unsigned int translateDrawingPrimitive(unsigned int aDrawPrimitive) = 0;



			//virtual void loadIdentityAndSetCamera(nau::scene::Camera& aCamera) = 0;


			/*Matrix Operations*/


			//virtual void unproject (nau::render::IRenderable &aRenderable, nau::scene::Camera& aCamera) = 0;



			//virtual void setFixedFunction (bool fixed) = 0;


			virtual void flush (void) = 0;



			/* Debug Operations */
			
			//virtual void renderBoundingVolume (const nau::geometry::IBoundingVolume* aBoundingVolume) = 0;





			

			virtual nau::math::vec3 readpixel (int x, int y) = 0;
	
			/* Frustum Operation */ 
			
			virtual void activateUserClipPlane (unsigned int aClipPlane) = 0;
			virtual void setUserClipPlane(unsigned int aClipPlane, double *plane) = 0;
			virtual void deactivateUserClipPlane(unsigned int  aClipPlane) = 0;

			/* Picking Operations */

			/* Material Operations */

			//! Set all color properties
			
			virtual ~IRenderer() {}

			/*  Shaders */

		};
	};
};

#endif
