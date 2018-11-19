#ifndef INAU_H
#define INAU_H

#pragma warning( disable: 4290)

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif


#include "nau/config.h"
#include "nau/attribute.h"
#include "nau/attributeValues.h"

#include "nau/event/eventManager.h"
#include "nau/event/iListener.h"
#include "nau/material/material.h"
#include "nau/math/vec3.h"
#include "nau/math/vec4.h"
#include "nau/physics/physicsManager.h"
#include "nau/render/pipeline.h"
#include "nau/render/pass.h"
#include "nau/render/viewport.h"
#include "nau/render/renderManager.h"
#include "nau/resource/resourceManager.h"
#include "nau/scene/iScene.h"
#include "nau/scene/camera.h"
#include "nau/scene/light.h"

#include <iostream>

//I know Peter, but you'll see that this may come in handy ;)

//#ifdef _WINDLL
//#define NAU INau::GetInterface()
//#else
#define NAU Nau::GetInstance()
//#endif

#define RENDERER NAU->getRenderer()
#define RENDERMANAGER NAU->getRenderManager()
#define MATERIALLIBMANAGER NAU->getMaterialLibManager()
#define RESOURCEMANAGER NAU->getResourceManager()
//#define EVENTMANAGER NAU->getEventManager()
#define APISupport NAU->getAPISupport()


namespace nau {
	
	class INau : public IListener, public AttributeValues
	{

	public:

		static INau *GetInterface();
		static void SetInterface(INau *n);

		enum RenderFlags {
			BOUNDING_BOX_RENDER_FLAG,
			PROFILE_RENDER_FLAG,
			COUNT_RENDER_FLAGS
		};

		virtual void setProjectName(std::string name) = 0;
		virtual const std::string &getProjectName() = 0;

		virtual std::string getProjectFolder() = 0;

#if NAU_LUA == 1
		virtual void callLuaScript(std::string name) = 0;
		virtual void initLuaScript(std::string file, std::string name) = 0;
		virtual bool callLuaTestScript(std::string name) = 0;
		virtual void compileLuaScripts() = 0;

#endif

		// Global gets and sets
		// note: gets and set perform no validation
		// if in doubt call validate first

		// Fully validate - context must refer to an existing object
		//virtual bool validateAttribute(std::string type, std::string context, std::string component) = 0;
		// Only validates the existence of the component in a particular type/context of object
		virtual bool validateShaderAttribute(std::string type, std::string context, std::string component) = 0;
		// returns Enums::COUNT_DATATYPE if attribute does not exist
		virtual Enums::DataType getAttributeDataType(std::string type, std::string context, std::string component) = 0;
		virtual bool setAttributeValue(std::string type, std::string context,
				 std::string component, int number,
				 Data *values) = 0;
		virtual void *getAttributeValue(std::string type, std::string context,
			std::string component, int number=0) = 0;

		virtual AttributeValues *createObject(const std::string &objType, const std::string &name) = 0;
		virtual std::unique_ptr<Attribute> &getAttribute(const std::string &type, const std::string &component) = 0;

		virtual AttributeValues *getObjectAttributes(const std::string &type, const std::string &context, int number=0) = 0;
		virtual AttributeValues *getCurrentObjectAttributes(const std::string &context, int number = 0) = 0;
		virtual bool validateObjectType(const std::string & type) = 0;
		virtual void getValidObjectTypes(std::vector<std::string>* v) = 0;
		virtual void getValidObjectNames(const std::string & type, std::vector<std::string>* v) = 0;
		virtual bool validateObjectName(const std::string & type, const std::string & v) = 0;
		virtual bool validateObjectContext(const std::string & type, const std::string & context) = 0;
		virtual bool validateObjectComponent(const std::string & type, const std::string & component) = 0;
		virtual void getValidObjectComponents(const std::string &type, std::vector<std::string>* v) = 0;

		// Profile Reset
		virtual void setProfileResetRequest() = 0;
		virtual bool getProfileResetRequest() = 0;

		virtual void registerAttributes(std::string s, AttribSet *attrib) = 0;
		virtual bool validateUserAttribType(std::string s) = 0;
		virtual bool validateUserAttribName(std::string context, std::string name) = 0;
		virtual AttribSet *getAttribs(std::string context) = 0;
		virtual void getObjTypeList(std::vector<std::string> *) = 0;

		virtual nau::scene::Camera *getActiveCamera() = 0;

		virtual unsigned int getWindowHeight() = 0;
		virtual unsigned int getWindowWidth() = 0;

		// Viewports
		virtual void setActiveCameraName(const std::string &aCamName) = 0;
		virtual std::shared_ptr<Viewport> getDefaultViewport () = 0;
		virtual void setWindowSize(unsigned int width, unsigned int height) = 0;
		virtual float getDepthAtCenter() = 0;


		/* Managers */
		virtual nau::render::RenderManager* getRenderManager (void) = 0;
		virtual nau::render::IRenderer *getRenderer(void) = 0;
		virtual nau::resource::ResourceManager* getResourceManager (void) = 0;
		virtual nau::material::MaterialLibManager* getMaterialLibManager (void) = 0;
		virtual nau::event_::EventManager* getEventManager (void) = 0;
		virtual nau::physics::PhysicsManager *getPhysicsManager() = 0;

		virtual IAPISupport* getAPISupport(void) = 0;

		virtual void loadAsset(std::string aFilename, std::string sceneName, std::string params = "") throw (std::string) = 0;
		virtual void writeAssets(std::string fileType, std::string aFilename, std::string sceneName) = 0;

		virtual bool getTraceStatus() = 0;
		virtual void setRenderFlag(RenderFlags aFlag, bool aState) = 0;
		virtual bool getRenderFlag(RenderFlags aFlag) = 0;

	protected:
		~INau(void) {};
		INau() {};

		static INau *Interface;
	};
};


#endif //INAU_H
