#ifndef NAU_H
#define NAU_H

#pragma warning( disable: 4290)

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <nau/errors.h>
#include <nau/config.h>
#include <nau/event/eventManager.h>
#include <nau/event/ilistener.h>
#include <nau/material/materiallibmanager.h>
#include <nau/math/vec3.h>
#include <nau/math/vec4.h>
#include <nau/render/pipeline.h>
#include <nau/render/pass.h>
#include <nau/render/viewport.h>
#include <nau/render/rendermanager.h>
#include <nau/resource/resourcemanager.h>
#include <nau/scene/iscene.h>
#include <nau/scene/camera.h>
#include <nau/scene/light.h>
#include <nau/world/iworld.h>

#include <cmath>
#include <iostream>

using namespace nau;

//I know Peter, but you'll see that this may come in handy ;)
#define NAU Nau::getInstance()
#define RENDERER NAU->getRenderManager()->getRenderer()
#define RENDERMANAGER NAU->getRenderManager()
#define MATERIALLIBMANAGER NAU->getMaterialLibManager()
#define RESOURCEMANAGER NAU->getResourceManager()
#define EVENTMANAGER NAU->getEventManager()

namespace nau {
	
	const double NO_TIME = -1.0f;

	class Nau : public IListener
	{

	public:		

		typedef enum RenderFlags {
						BOUNDING_BOX_RENDER_FLAG, 
						PROFILE_RENDER_FLAG, 
						COUNT_RENDER_FLAGS
		};

		static nau::Nau* create (void);
		static nau::Nau* getInstance (void);
		bool init(bool context, std::string aConfigFile = "");
		std::string &getName();

		// Attributes
		void registerAttributes(std::string s, AttribSet *attrib);
		bool validateUserAttribContext(std::string s);
		bool validateUserAttribName(std::string context, std::string name);
		AttribSet *getAttribs(std::string context);
		void deleteUserAttributes();


		void eventReceived(const std::string &sender, const std::string &eventType, IEventData *evt);

		void setActiveCameraName(const std::string &aCamName);
		nau::scene::Camera *getActiveCamera();

		float getDepthAtCenter();

		//void setProfileMaterial(std::string aMaterial);

		nau::world::IWorld& getWorld (void);

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
		unsigned long int getFrameCount();

		void loadAsset (std::string aFilename, std::string sceneName, std::string params = "") throw (std::string);
		void writeAssets (std::string fileType, std::string aFilename, std::string sceneName);

		void setWindowSize (float width, float height);
		float getWindowHeight();
		float getWindowWidth();
		nau::render::Viewport* createViewport (const std::string &name, nau::math::vec4 &bgColor);
		nau::render::Viewport* createViewport (const std::string &name);
		nau::render::Viewport* getViewport (const std::string &name);
		nau::render::Viewport* getDefaultViewport ();
		std::vector<std::string> *getViewportNames();

		bool reload (void);

		void sendKeyToEngine (char keyCode); /***Change this in to a register system. The sub-system register as a particular key receiver*/

		void enablePhysics (void);
		void disablePhysics (void); 

		int picking (int x, int y, std::vector<nau::scene::SceneObject*> &objects, nau::scene::Camera &aCamera);

		//void addAnimation (std::string animationName, nau::animation::IAnimation *anAnimation);
		//nau::animation::IAnimation *getAnimation (std::string animationName);
		
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

		/* Render Flags */
		void setRenderFlag(RenderFlags aFlag, bool aState);
		bool getRenderFlag(RenderFlags aFlag);

		~Nau(void);
		void clear();

		//State
		void loadStateXMLFile(std::string file);
		std::vector<std::string> getStateEnumNames();
		std::string getState(std::string enumName);
	private:
		Nau();

		std::string m_Name;
		unsigned long int m_FrameCount;

		/*
		Attributes
		*/
		typedef AttribSet *AttribSetPointer;
		AttribSetPointer a;
		std::map<std::string, AttribSetPointer> m_Attributes;
		/*
		 * Rendering Flags
		 */
		std::vector<bool> m_RenderFlags;
		//nau::material::Material *m_ProfileMaterial; 
		bool m_UseTangents, m_UseTriangleIDs, m_CoreProfile;
		/*
		 * Managers
		 */
		nau::render::RenderManager *m_pRenderManager;
		nau::resource::ResourceManager *m_pResourceManager;
		nau::material::MaterialLibManager *m_pMaterialLibManager;
		nau::event_::EventManager *m_pEventManager;

		/*
		 * Members
		 */
		std::string m_ActiveCameraName;
		float m_WindowWidth, m_WindowHeight;
		nau::world::IWorld *m_pWorld;
		std::map <std::string, nau::render::Viewport*> m_vViewports;
		nau::render::Viewport *m_Viewport;
		//std::map <std::string, nau::animation::IAnimation*> m_Animations;
		IState *m_DefaultState;

		bool m_Inited;
		bool m_Physics;
		
		double m_CurrentTime;
		double m_LastFrameTime;

		double CLOCKS_PER_MILISEC;
		double INV_CLOCKS_PER_MILISEC;

		// different perspective and camera position depending on whether
		// the model is unitized
		void Nau::loadFilesAndFoldersAux(char *sceneName, bool unitize);

		int loadedScenes;

		bool isFrameBegin;


	};
};

#endif //NAU_H
