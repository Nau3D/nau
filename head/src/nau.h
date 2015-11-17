#ifndef NAU_H
#define NAU_H

#pragma warning( disable: 4290)

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "iNau.h"
#include "nau/config.h"
#include "nau/attribute.h"
#include "nau/attributeValues.h"

#include "nau/errors.h"
#include "nau/event/eventManager.h"
#include "nau/event/ilistener.h"
#include "nau/material/materialLibManager.h"
#include "nau/math/vec3.h"
#include "nau/math/vec4.h"
#include "nau/render/pipeline.h"
#include "nau/render/pass.h"
#include "nau/render/viewport.h"
#include "nau/render/renderManager.h"
#include "nau/resource/resourceManager.h"
#include "nau/scene/iScene.h"
#include "nau/scene/camera.h"
#include "nau/scene/light.h"
#include "nau/world/iWorld.h"

#ifdef NAU_LUA
extern "C" {
#include <lua/lua.h>
#include <lua/lauxlib.h>
#include <lua/lualib.h>
}
#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <iostream>

using namespace nau;

//I know Peter, but you'll see that this may come in handy ;)
//#define NAU Nau::GetInstance()
//#define RENDERER NAU->getRenderManager()->getRenderer()
//#define RENDERMANAGER NAU->getRenderManager()
//#define MATERIALLIBMANAGER NAU->getMaterialLibManager()
//#define RESOURCEMANAGER NAU->getResourceManager()
//#define EVENTMANAGER NAU->getEventManager()
//#define APISupport NAU->getAPISupport()


namespace nau {
	
	const double NO_TIME = -1.0f;

	class Nau : public INau // , public IListener, public AttributeValues
	{

	public:		


		static nau::Nau* Create (void);
		static nau::Nau* GetInstance (void);
#ifdef _WINDLL
		static void SetInstance(Nau *inst);
#endif
		bool init(bool context, std::string aConfigFile = "");

		void setProjectName(std::string name);
		const std::string &getProjectName();


		// Lua Stuff
#ifdef NAU_LUA
		void initLua();
		void initLuaScript(std::string file, std::string name);
		void callLuaScript(std::string name);
		bool callLuaTestScript(std::string name);
#endif
		std::string &getName();

		// Global gets and sets
		// note: gets and set perform no validation
		// if in doubt call validate first

		// Fully validate - context must refer to an existing object
		bool validateAttribute(std::string type, std::string context, std::string component);
		// Only validates the existence of the component in a particular type/context of object
		bool validateShaderAttribute(std::string type, std::string context, std::string component);
		bool setAttribute(std::string type, std::string context,
				 std::string component, int number,
				 Data *values);
		void *getAttribute(std::string type, std::string context,
			std::string component, int number);
		AttributeValues *getObjectAttributes(std::string type, std::string context, int number=0);
		AttributeValues *getCurrentObjectAttributes(std::string context, int number = 0);

		// Attributes
		void registerAttributes(std::string s, AttribSet *attrib);
		bool validateUserAttribContext(std::string s);
		bool validateUserAttribName(std::string context, std::string name);
		AttribSet *getAttribs(std::string context);
		void deleteUserAttributes();
		std::vector<std::string> &getContextList();

		// Events
		void eventReceived(const std::string &sender, const std::string &eventType, IEventData *evt);


		void setActiveCameraName(const std::string &aCamName);
		nau::scene::Camera *getActiveCamera();

		float getDepthAtCenter();

		//void setProfileMaterial(std::string aMaterial);

		nau::world::IWorld& getWorld (void);

		void setTrace(int frames);
		bool getTraceStatus();
		// Executes the whole pipeline
		void step ();
		// Executes the next pass
		// only to be used when in paused mode
		void stepPass();
		// Executes the pipeline from the current pass to the end
		// only to be used when in paused mode
		void stepCompleteFrame();
		// executes n passes from the pipeline. It may loop.
		// only to be used when in paused mode
		void stepPasses(int n);

		void resetFrameCount();
//		unsigned long int getFrameCount();

		void loadAsset (std::string aFilename, std::string sceneName, std::string params = "") throw (std::string);
		void writeAssets (std::string fileType, std::string aFilename, std::string sceneName);

		void setWindowSize (unsigned int width, unsigned int height);
		unsigned int getWindowHeight();
		unsigned int getWindowWidth();

		// Viewports
		std::shared_ptr<Viewport> getDefaultViewport ();

		bool reload (void);

		void sendKeyToEngine (char keyCode); /***Change this in to a register system. The sub-system register as a particular key receiver*/
		void setClickPosition(int x, int y);

		void enablePhysics (void);
		void disablePhysics (void); 

		int picking (int x, int y, std::vector<nau::scene::SceneObject*> &objects, nau::scene::Camera &aCamera);

		/* Readers */
		void readModel (std::string fileName) throw (std::string);
		void appendModel(std::string fileName);
		void readProjectFile (std::string file, int *width, int *height);
		void Nau::readDirectory (std::string dirName);

		/* Managers */
		nau::render::RenderManager* getRenderManager (void);
		nau::resource::ResourceManager* getResourceManager (void);
		nau::material::MaterialLibManager* getMaterialLibManager (void);
		nau::event_::EventManager* getEventManager (void);
		nau::render::IRenderer *getRenderer(void);
		IAPISupport * getAPISupport(void);


		/* Render Flags */
		void setRenderFlag(RenderFlags aFlag, bool aState);
		bool getRenderFlag(RenderFlags aFlag);

		~Nau(void);
		void clear();


	private:
		Nau();

#ifdef NAU_LUA
		lua_State *m_LuaState;
#endif

		std::string m_ProjectName;
		float m_StartTime;
		int m_TraceFrames;
		bool m_TraceOn;

		std::string m_Name;

		/*
		Attributes
		*/
		typedef AttribSet *AttribSetPointer;
		AttribSetPointer a;
		std::map<std::string, AttribSet*> m_Attributes;
		/*
		 * Rendering Flags
		 */
		std::vector<bool> m_RenderFlags;
		bool m_UseTangents, m_UseTriangleIDs, m_CoreProfile;
		/*
		 * Managers
		 */
		nau::render::RenderManager *m_pRenderManager;
		nau::resource::ResourceManager *m_pResourceManager;
		nau::material::MaterialLibManager *m_pMaterialLibManager;
		nau::event_::EventManager *m_pEventManager;
		nau::render::IAPISupport *m_pAPISupport;

		/*
		 * Members
		 */
		std::string m_AppFolder;
		std::string m_ActiveCameraName;
		unsigned int m_WindowWidth, m_WindowHeight;
		nau::world::IWorld *m_pWorld;
		std::shared_ptr<Viewport> m_Viewport;
		int m_ClickX = 0, m_ClickY = 0;
		IState *m_DefaultState;

		bool m_Inited;
		bool m_Physics;
		
		//double m_CurrentTime;
		double m_LastFrameTime;

		double CLOCKS_PER_MILISEC;
		double INV_CLOCKS_PER_MILISEC;

		// different perspective and camera position depending on whether
		// the model is unitized
		void Nau::loadFilesAndFoldersAux(std::string sceneName, bool unitize);

		int loadedScenes;

		bool isFrameBegin;

		// this vector allows returning string vectors safely without
		// memory leaks
		std::vector<std::string> m_DummyVector;

	};
};

#endif //NAU_H
