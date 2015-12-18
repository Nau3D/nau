#ifndef IRENDERER_H
#define IRENDERER_H

#include <vector>
#include <string>


#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/enums.h"
#include "nau/geometry/frustum.h"
#include "nau/geometry/iBoundingVolume.h"
#include "nau/material/colorMaterial.h"
#include "nau/material/material.h"
#include "nau/material/materialGroup.h"
#include "nau/material/iProgram.h"
#include "nau/math/matrix.h"
#include "nau/math/vec4.h"
#include "nau/scene/camera.h"
#include "nau/scene/iScene.h"
#include "nau/scene/light.h"
#include "nau/scene/sceneObject.h"
#include "nau/util/tree.h"

using namespace nau;
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
			static std::map<std::string, IRenderable::DrawPrimitive> PrimitiveTypes;

			static AttribSet Attribs;

			MAT4_PROP(PROJECTION, 0);
			MAT4_PROP(MODEL, 1);
			MAT4_PROP(VIEW, 2);
			MAT4_PROP(TEXTURE, 3);
			MAT4_PROP(VIEW_MODEL, 4);
			MAT4_PROP(PROJECTION_VIEW_MODEL, 5);
			MAT4_PROP(PROJECTION_VIEW, 6);
			MAT4_PROP(TS05_PVM, 7);

			MAT3_PROP(NORMAL, 0);

			INT_PROP(TEXTURE_COUNT, 0);
			INT_PROP(LIGHT_COUNT, 1);

			UINT_PROP(INSTANCE_COUNT, 0);
			UINT_PROP(BUFFER_DRAW_INDIRECT, 1);
			UINT_PROP(FRAME_COUNT, 2);

			FLOAT_PROP(TIMER, 0);

			INT2_PROP(MOUSE_CLICK, 0);

			BOOL_PROP(DEBUG_DRAW_CALL, 0);

			static int MaxTextureUnits;
			static int MaxColorAttachments;

			typedef enum {
				FRONT,
				BACK,
			} Face;


			const static int PRIMITIVE_TYPE_COUNT = 8;

			typedef enum {
				COLOR_BUFFER = 0x01,
				DEPTH_BUFFER = 0x02,
				STENCIL_BUFFER = 0x04
			} FrameBuffer;

			virtual bool init() = 0;

			//virtual void *getProp(unsigned int prop, Enums::DataType dt) = 0;
			virtual const mat4 &getPropm4(Mat4Property prop) = 0;
			virtual const mat3 &getPropm3(Mat3Property prop) = 0;
			virtual float getPropf(FloatProperty prop);
			void setPropb(BoolProperty prop, bool value);
			
			// SHADER DEBUG INFO
		protected:
			nau::util::Tree m_ShaderDebugTree;
		public:
			nau::util::Tree *getShaderDebugTree();

			// API SUPPORT
		public:
			bool primitiveTypeSupport(std::string primitive);


			// TRACE API
		public:
			virtual int setTrace(int) = 0;

			// ATOMIC COUNTERS 

		public:
			/// Number of Atomic Counters and
			unsigned int m_AtomicCount = 0;
			/// name of atomic counters
			/// <bufferName, offset> -> atomic label
			std::map<std::pair<std::string,unsigned int>, std::string> m_AtomicLabels;

			/// add an atomic counter
			void addAtomic(std::string buffer, unsigned int offset, std::string name);

			/// get atomic counter values
			virtual std::vector<unsigned int> &getAtomicCounterValues() = 0;

		protected:
			/// flag indicating if Atomic Buffer is created
			bool m_AtomicBufferPrepared = false;

		// LIGHTS

		public:
			/// returns the number of active lights
			virtual unsigned int getLightCount() = 0;
			/// returns light index i
			virtual std::shared_ptr<Light> &getLight(unsigned int i) = 0;
			/// adds a light at the next free index
			virtual bool addLight(std::shared_ptr<Light> &l) = 0;
			/// removes all lights
			virtual void removeLights() = 0;


		// CAMERA

		public:
			/// sets viewport
			virtual void setViewport(std::shared_ptr<Viewport>) = 0;
			/// returns currrent viewport
			virtual std::shared_ptr<Viewport> getViewport() = 0;
			/// set the camera
			virtual void setCamera(std::shared_ptr<Camera> &aCamera) = 0;
			/// returns current camera
			virtual std::shared_ptr<Camera> &getCamera() = 0;


		// COUNTERS

		public:
			typedef enum {
				TRIANGLE_COUNTER
			} Counters;

			/// resets all counters
			virtual void resetCounters(void) = 0;
			/// returns counter value
			virtual unsigned int getCounter(Counters c) = 0;
			virtual unsigned int getNumberOfPrimitives(MaterialGroup *m) = 0;



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
			///// returns a float array with matrix values
			//virtual const float *getMatrix(MatrixType aType) = 0;
			/// pushes matrix into matrix stack
			virtual void pushMatrix(MatrixMode mode) = 0;
			/// pops matrix into matrix stack
			virtual void popMatrix(MatrixMode mode) = 0;
			/// compose Current = Current * aTransform
			virtual void applyTransform(MatrixMode mode, const nau::math::mat4 &aTransform) = 0;
			/// compose Current = Current * translate(vec)
			virtual void translate(MatrixMode mode, nau::math::vec3 &aVec) = 0;
			/// compose Current = Current * scale(vec)
			virtual void scale(MatrixMode mode, nau::math::vec3 &aVec) = 0;
			/// compose Current = Current * rotate(angle,axis)
			virtual void rotate(MatrixMode mode, float angle, nau::math::vec3 &axis) = 0;


		// MATERIAL (color, state, shaders and textures)

		public:
			// color
			virtual void setMaterial(nau::material::ColorMaterial &mat) = 0;
			virtual void setMaterial(vec4 &diffuse, vec4 &ambient, vec4 &emission, vec4 &specular, float shininess) = 0;
			virtual ColorMaterial *getMaterial() = 0;

			// state
			virtual void setState(IState *aState) = 0;
			virtual void setDefaultState() = 0;
			virtual IState *getState() = 0;

			/// shaders
			virtual void setShader(IProgram *aShader) = 0;
			/// returns the attribute location of a uniform var
			virtual int getAttribLocation(std::string &name) = 0;

			// image textures
			virtual void addImageTexture(unsigned int aTexUnit, IImageTexture *t) = 0;
			virtual void removeImageTexture(unsigned int aTexUnit) = 0;
			virtual unsigned int getImageTextureCount() = 0;
			virtual IImageTexture* getImageTexture(unsigned int unit) = 0;

			// textures 
			virtual void setActiveTextureUnit(unsigned int aTexUnit) = 0;
			virtual void addTexture(MaterialTexture *t) = 0;
			virtual void removeTexture(unsigned int aTexUnit) = 0;
			virtual MaterialTexture *getMaterialTexture(int unit) = 0;
			virtual ITexture *getTexture(int unit) = 0;
			virtual void resetTextures(const std::map<int, MaterialTexture *> &textures) = 0;

			// FRAMEBUFFER OPS

		public:

			virtual void clearFrameBuffer (unsigned int b) = 0;
			virtual void prepareBuffers(Pass *p) = 0;

			virtual void setDepthClamping(bool b) = 0;
			virtual void colorMask(bool r, bool g, bool b, bool a) = 0;
			virtual void saveScreenShot() = 0;


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
			virtual void drawGroup(std::shared_ptr<nau::material::MaterialGroup> aMaterialGroup) = 0;
			virtual void setCullFace(Face aFace) = 0;
			virtual void dispatchCompute(int dimX, int dimY, int dimZ) = 0;

			// RENDER ATTRIBS

			virtual void saveAttrib(RendererAttributes aAttrib) = 0;
			virtual void restoreAttrib() = 0;


			// MISC

			virtual float getDepthAtPoint(int x, int y) = 0;
			virtual unsigned int translateDrawingPrimitive(unsigned int aDrawPrimitive) = 0;
			virtual void flush (void) = 0;
			virtual nau::math::vec3 readpixel (int x, int y) = 0;
			
			virtual ~IRenderer() {}

		};
	};
};

#endif
