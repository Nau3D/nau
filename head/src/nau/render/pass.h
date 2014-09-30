#ifndef PASS_H
#define PASS_H

#include <vector>
#include <string>
#include <map>

#include <nau/geometry/boundingbox.h>
#include <nau/geometry/quad.h>
#include <nau/material/materialid.h>
#include <nau/scene/camera.h>
#include <nau/scene/iscene.h>
#include <nau/scene/sceneobject.h>
#include <nau/render/rendertarget.h>
#include <nau/render/texture.h>
#include <nau/scene/geometryobject.h>

#include <nau/event/eventManager.h>
#include <nau/event/ilistener.h>
#include <nau/event/eventString.h>
#include <nau/attributeValues.h>
#include <nau/attribute.h>

namespace nau
{
	namespace render
	{
		class Pass : public IListener, public AttributeValues {

		public:

			BOOL_PROP(COLOR_CLEAR, 0);
			BOOL_PROP(COLOR_ENABLE, 1);
			BOOL_PROP(DEPTH_CLEAR, 2);
			BOOL_PROP(DEPTH_ENABLE, 3);
			BOOL_PROP(DEPTH_MASK, 4);
			BOOL_PROP(DEPTH_CLAMPING, 5);
			BOOL_PROP(STENCIL_CLEAR, 6);
			BOOL_PROP(STENCIL_ENABLE, 7);

			FLOAT_PROP(DEPTH_CLEAR_VALUE, 0);
			FLOAT_PROP(STENCIL_CLEAR_VALUE, 1);

			FLOAT4_PROP(COLOR_CLEAR_VALUE, 0);

			INT_PROP(STENCIL_OP_REF, 0);

			UINT_PROP(STENCIL_OP_MASK, 0);

			ENUM_PROP(STENCIL_FUNC, 0);
			ENUM_PROP(STENCIL_FAIL, 1);
			ENUM_PROP(STENCIL_DEPTH_FAIL, 2);
			ENUM_PROP(STENCIL_DEPTH_PASS, 3);
			ENUM_PROP(DEPTH_FUNC, 4);

			static AttribSet Attribs;

			void setPropb(BoolProperty prop, bool value);
			//bool getPropb(IRenderer::BoolProps prop);

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
				RUN_ALWAYS,

			} RunMode;

			typedef enum {
				LESS, NEVER, ALWAYS, LEQUAL,
				EQUAL, GEQUAL, GREATER, NOT_EQUAL
			} StencilFunc;

		protected:
			std::string p_Empty;
			void initVars();

			static bool Init();
			static bool Inited;

			//std::vector<bool> m_BoolProp;

			std::string m_ClassName;
			std::string m_Name;
			std::string m_CameraName;
			std::vector<std::string> m_SceneVector;
			std::map<std::string, nau::material::MaterialID> m_MaterialMap;
			nau::render::Viewport *m_pViewport;
			nau::render::Viewport *m_pRestoreViewport;
						
			nau::render::RenderTarget *m_RenderTarget;
			//nau::render::Texture* m_TexId[MAXFBOs+1];	

			//std::map<std::string, float> m_Paramf;
			//std::map<std::string, int> m_Parami;
			//std::map<std::string, Enums::DataType> m_ParamType;

			//vec4 m_ColorClearValue;

			//float m_DepthClearValue;
			//int m_DepthFunc;

			//int m_StencilClearValue;
			//int m_StencilMaskValue;
			//StencilOp m_Stencilsfail, m_Stencildfail, m_Stencildpass;
			//StencilFunc m_StencilFunc;
			//int m_StencilOpRef;
			//unsigned int m_StencilOpMask;

			int m_RTSizeWidth; // size of render targets
			int m_RTSizeHeight;

			//int m_Depth; 
			//int m_Color; // number of render targets

			bool m_UseRT;

			typedef enum {REMAP_DISABLED, REMAP_TO_ONE, REMAP_TO_LIBRARY} RemapMode;
			
			RemapMode m_RemapMode;

			std::vector<std::string> m_Lights;

			//std::map<std::string, std::string> m_Params;
			//nau::render::Texture* m_Inputs[8];
			//nau::render::RenderTarget* m_DepthBuffer;
			//bool m_DoColorClear;
			//bool m_DoDepthClear;
			//bool m_DepthMask;
			//bool m_DoStencilClear;
			//std::vector<std::string> m_ParamNames;
			//std::vector<TYPES> m_ParamTypes;

		public:
			Pass (const std::string &passName);
			virtual ~Pass();

			void eventReceived(const std::string &sender, const std::string &eventType, IEventData *evtData);

			const std::string &getClassName();

			//virtual const std::map<std::string, float> &getParamsf();
			//virtual void setParam(const std::string &name, const float value);
			//virtual void setParam(const std::string &name, const int value);
		
			//virtual float *getParamf(const std::string &name);
			//virtual int *getParami(const std::string &name);
			//virtual int getParamType(const std::string &name);

			std::string &getName (void);



			virtual void addScene (const std::string &sceneName);
			bool hasScene(const std::string &name);
			void removeScene(const std::string &name);

			void updateMaterialMaps(const std::string &sceneName);

			const std::vector<std::string>& getScenesNames (void);

			virtual void setCamera (const std::string &cameraName);
			const std::string& getCameraName (void);

			void setViewport (nau::render::Viewport *aViewport);
			nau::render::Viewport *getViewport();
			
			void setColorClearValue(float r, float g, float b, float  a);

			void setDepthClearValue(float value);
			void setDepthFunc(int f);

			void setStencilClearValue(float value);
			//void setStencilMaskValue(int i);
			void setStencilFunc(Pass::StencilFunc f, int ref, unsigned int mask);
			void setStencilOp(	Pass::StencilOp sfail, 
							Pass::StencilOp dfail, 
							Pass::StencilOp dpass);



			virtual void prepare (void);
			virtual void restore (void);
			virtual bool renderTest (void);
			virtual void doPass (void);

			/*Lights*/
			virtual void addLight (const std::string &name);
			bool hasLight(const std::string &name);
			void removeLight(const std::string &name);
				
			/*Materials*/
			const std::map<std::string, nau::material::MaterialID> &getMaterialMap();
			void remapMaterial (const std::string &originMaterialName, 
								const std::string &materialLib, 
								const std::string &destinyMaterialName);
			void remapAll (const std::string &materialLib, 
								const std::string &destinyMaterialName);
			void remapAll (const std::string &targetLibrary);

			void materialNamesFromLoadedScenes (std::vector<std::string> &materials);

			/*Rendertargets*/
			nau::render::RenderTarget* getRenderTarget (void);
			virtual void setRenderTarget (nau::render::RenderTarget* rt);
			void enableRenderTarget(bool b);
			bool isRenderTargetEnabled();

			bool hasRenderTarget();
		
		protected:
			virtual void setupCamera (void);
			void restoreCamera (void);
			void setupLights (void);

			void prepareBuffers();

			void setRTSize (int width, int height);
		/***MARK***/ //Maybe this should be moved to the BoundingBox class
			nau::geometry::BoundingBox getBoundingBox (std::vector<nau::scene::SceneObject*> &sceneObjects);


		};
	};
};
#endif





	
	
