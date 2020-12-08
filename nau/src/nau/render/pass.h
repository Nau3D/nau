/*
Class Pass

A pass is the building block of a pipeline.
Passes may have geometry (scenes), cameras, lights, viewports
and other elements. 

This class implements the basic pass from which all pass variations 
should inherit.


https://github.com/Nau3D

*/

#ifndef PASS_H
#define PASS_H


#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/event/eventManager.h"
#include "nau/event/eventString.h"
#include "nau/event/iListener.h"
#include "nau/geometry/boundingBox.h"
#include "nau/geometry/quad.h"
#include "nau/material/iBuffer.h"
#include "nau/material/materialId.h"
#include "nau/material/iTexture.h"
#include "nau/render/passProcessItem.h"
#include "nau/render/iRenderTarget.h"
#include "nau/scene/camera.h"
#include "nau/scene/geometricObject.h"
#include "nau/scene/iScene.h"
#include "nau/scene/sceneObject.h"


#include <map>
#include <string>
#include <vector>

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

namespace nau
{
	namespace render
	{
		class Pass : public IListener, public AttributeValues {

			friend class PassFactory;
		public:
			// from passOptixPrime
			INT_PROP(RAY_COUNT, 201);

			// Pass properties

			BOOL_PROP(COLOR_CLEAR, 0);
			BOOL_PROP(COLOR_ENABLE, 1);
			BOOL_PROP(DEPTH_CLEAR, 2);
			BOOL_PROP(DEPTH_ENABLE, 3);
			BOOL_PROP(DEPTH_MASK, 4);
			BOOL_PROP(DEPTH_CLAMPING, 5);
			BOOL_PROP(STENCIL_CLEAR, 6);
			BOOL_PROP(STENCIL_ENABLE, 7);

			FLOAT_PROP(DEPTH_CLEAR_VALUE, 0);

			FLOAT4_PROP(COLOR_CLEAR_VALUE, 0);

			INT_PROP(STENCIL_OP_REF, 0);
			INT_PROP(STENCIL_CLEAR_VALUE, 1);
			INT_PROP(RAYS_PER_PIXEL, 2);
			INT_PROP(MAX_DEPTH, 3);

			UINT_PROP(STENCIL_OP_MASK, 0);
			UINT_PROP(INSTANCE_COUNT, 1);
			UINT_PROP(BUFFER_DRAW_INDIRECT, 2);

			ENUM_PROP(STENCIL_FUNC, 0);
			ENUM_PROP(STENCIL_FAIL, 1);
			ENUM_PROP(STENCIL_DEPTH_FAIL, 2);
			ENUM_PROP(STENCIL_DEPTH_PASS, 3);
			ENUM_PROP(DEPTH_FUNC, 4);
			ENUM_PROP(RUN_MODE, 5);
			ENUM_PROP(TEST_MODE, 6);

			// from pass compute
			UINT_PROP(DIM_X, 101);
			UINT_PROP(DIM_Y, 102);
			UINT_PROP(DIM_Z, 103);

			STRING_PROP(CAMERA, 0);

			typedef enum {
				KEEP,
				ZERO,
				REPLACE,
				INCR,
				INCR_WRAP,
				DECR,
				DECR_WRAP,
				INVERT
			} StencilOp;

			typedef enum {
				DONT_RUN,
				RUN_ALWAYS,
				SKIP_FIRST_FRAME,
				RUN_ONCE,
				RUN_EVEN,
				RUN_ODD
			} RunMode;

			typedef enum {
				RUN_IF,
				RUN_WHILE
			} TestMode;

			typedef enum {
				LESS, NEVER, ALWAYS, LEQUAL,
				EQUAL, GEQUAL, GREATER, NOT_EQUAL
			} StencilFunc;

			static nau_API AttribSet Attribs;
			static nau_API AttribSet &GetAttribs();

			nau_API  virtual ~Pass();

			nau_API  static std::shared_ptr<Pass> Create(const std::string &name);

			nau_API void eventReceived(const std::string &sender, const std::string &eventType,
				const std::shared_ptr<IEventData> &evt);

			nau_API const std::string &getClassName();
			nau_API std::string &getName (void);

			//
			// LUA SCRIPTS
			//
			nau_API void setTestScript(std::string file, std::string name);
			//bool testScript();
			//
			// PRE POST PROCESS
			//
			nau_API void addPreProcessItem(PassProcessItem *pp);
			nau_API void addPostProcessItem(PassProcessItem *pp);

			nau_API PassProcessItem * getPreProcessItem(unsigned int i);
			nau_API PassProcessItem * getPostProcessItem(unsigned int i);

			nau_API void executePreProcessList();
			nau_API void executePostProcessList();

			//
			// RENDER TEST
			//
			nau_API void setMode(RunMode value);
			// 
			nau_API bool renderTest (void);

			// 
			// These are the three functions that are commonly 
			// redefined in subclasses

			// executes preparation code PRIOR to renderTest
			nau_API virtual void prepare (void);
			// executes only if render test is true
			nau_API virtual void doPass (void);
			// executes after doPass, regardless of render test
			nau_API virtual void restore (void);

			//
			// VIEWPORTS
			//
			nau_API void setViewport (int i, std::shared_ptr<Viewport>);
			nau_API void addViewport(std::shared_ptr<Viewport>);
			nau_API std::shared_ptr<Viewport> getViewport();

			//
			// LIGHTS
			//
			nau_API virtual void addLight (const std::string &name);
			nau_API bool hasLight(const std::string &name);
			nau_API void removeLight(const std::string &name);

			//
			// MATERIAL MAPS
			//
			nau_API void updateMaterialMaps(const std::string &sceneName);
			nau_API const std::map<std::string, nau::material::MaterialID> &getMaterialMap();
			nau_API void remapMaterial (const std::string &originMaterialName,
								const std::string &materialLib, 
								const std::string &destinyMaterialName);
			nau_API void remapAll (const std::string &materialLib,
								const std::string &destinyMaterialName);
			nau_API void remapAll (const std::string &targetLibrary);

			nau_API void materialNamesFromLoadedScenes (std::vector<std::string> &materials);

			//
			// RENDER TARGETS
			//
			nau_API nau::render::IRenderTarget* getRenderTarget (void);
			nau_API virtual void setRenderTarget (nau::render::IRenderTarget* rt);
			nau_API void enableRenderTarget(bool b);
			nau_API bool isRenderTargetEnabled();
			nau_API bool hasRenderTarget();

			//
			// CAMERAS
			//
			nau_API virtual void setCamera (const std::string &cameraName);
			nau_API const std::string& getCameraName (void);

			//
			// STENCIL
			//
			nau_API void setStencilClearValue(unsigned int value);
			nau_API void setStencilFunc(Pass::StencilFunc f, int ref, unsigned int mask);
			nau_API void setStencilOp(	Pass::StencilOp sfail,
							Pass::StencilOp dfail, 
							Pass::StencilOp dpass);

			//
			// DEPTH AND COLOR
			//
			nau_API void setDepthClearValue(float value);
			nau_API void setDepthFunc(int f);
			//void setColorClearValue(float r, float g, float b, float  a);

			//
			// SCENES
			//
			nau_API virtual void addScene (const std::string &sceneName);
			nau_API bool hasScene(const std::string &name);
			nau_API void removeScene(const std::string &name);
			nau_API const std::vector<std::string>& getScenesNames (void);

			//
			nau_API void setBufferDrawIndirect(std::string s);

			// -----------------------------------------------------------------
			//		PRE POST SCRIPTS
			// -----------------------------------------------------------------
			nau_API void setPreScript(std::string file, std::string name);
			nau_API void setPostScript(std::string file, std::string name);
			nau_API void callPreScript();
			nau_API void callPostScript();


		protected:

			nau_API Pass(const std::string &passName);
			// BUFFER DRAW INDIRECT
			IBuffer *m_BufferDrawIndirect = NULL;

			// LUA SCRIPTS
			std::string m_TestScriptFile, m_TestScriptName;
			std::string m_PreScriptFile, m_PreScriptName,
				m_PostScriptFile, m_PostScriptName;
			void callScript(std::string &name);

			// PRE POST PROCESS
			std::vector <PassProcessItem *> m_PreProcessList;
			std::vector <PassProcessItem *> m_PostProcessList;

			// CAMERAS
			nau_API virtual void setupCamera (void);
			nau_API void restoreCamera (void);
			// LIGHTS
			nau_API void setupLights (void);
			// called in prepare()
			nau_API void prepareBuffers();

			nau_API void setRTSize (uivec2 &v);

			// init class variables
			void initVars();

			// Init the attribute set
			static bool Init();
			static bool Inited;

			// pass class name, see passFactory.cpp for possible values
			std::string m_ClassName;
			// pass name
			std::string m_Name;
			//std::string m_CameraName;
			std::vector<std::string> m_Lights;
			std::vector<std::string> m_SceneVector;

			// VIEWPORTS
			bool m_ExplicitViewport;
			// used to temporarily store the camera viewport when the pass has an explicit viewport
			std::shared_ptr<Viewport> m_RestoreViewport;
			std::vector<std::shared_ptr<Viewport>> m_Viewport;

			nau::render::IRenderTarget *m_RenderTarget;
			// size of render targets
			int m_RTSizeWidth;
			int m_RTSizeHeight;

			bool m_UseRT;

			std::map<std::string, nau::material::MaterialID> m_MaterialMap;
			typedef enum {
				REMAP_DISABLED,
				REMAP_TO_ONE,
				REMAP_TO_LIBRARY
			} RemapMode;

			RemapMode m_RemapMode;

			int m_IntDummy;

		};
	};
};
#endif





	
	
