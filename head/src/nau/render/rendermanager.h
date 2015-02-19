#ifndef RENDERMANAGER_H
#define RENDERMANAGER_H


#include <vector>
#include <string>
#include <map>

#include <nau/render/pipeline.h>
#include <nau/render/irenderer.h>
#include <nau/render/irenderqueue.h>
#include <nau/render/viewport.h>
#include <nau/scene/camera.h>
#include <nau/scene/iscene.h>
#include <nau/scene/light.h>
#include <nau/scene/sceneobject.h>

namespace nau 
{
	namespace render 
	{
/**
* Responsible for the managment of pipelines, cameras, lights
* scenes, and scene objects. It also manages the renderqueue and the renderer
*/
		class RenderManager
		{
		private:
			IRenderer* m_pRenderer;
			IRenderQueue* m_pRenderQueue;
			std::map<std::string, Pipeline*> m_Pipelines;
			std::map<std::string, nau::scene::Camera*> m_Cameras;
			std::map<std::string, nau::scene::Light*> m_Lights;
			std::map<std::string, nau::scene::IScene*> m_Scenes;
			std::vector<nau::scene::SceneObject*> m_SceneObjects;
			std::map <std::string, nau::render::Viewport*> m_Viewports; 
			Pipeline *m_ActivePipeline;

		public:

			//! Required to get the engine started
			bool init();
			//! Cleans up memory
			void clear();

			//! Returns the actual renderer
			IRenderer* getRenderer (void);
			//! Renders the active pipeline
			unsigned char renderActivePipeline();
			void renderActivePipelineNextPass();

			// OCTREE STUFF
			//! Creates an octree for every OctreeScene
			void buildOctrees();

			//! Create VBOs for every IScene, erases all vertex data, except vertex coordinates 
			void compile();
			
			// VIEWPORTS
			nau::render::Viewport* createViewport(const std::string &name, nau::math::vec4 &bgColor);
			nau::render::Viewport* createViewport(const std::string &name);
			nau::render::Viewport* getViewport(const std::string &name);
			std::vector<std::string> *getViewportNames();
			bool hasViewport(const std::string &name);


			// PIPELINES
			//! Checks if a given named pipeline exists
			bool hasPipeline (const std::string &pipelineName);
			//! Returns a pointer to the named pipeline
			Pipeline* getPipeline (const std::string &pipelineName);
			//! Returns a pointer to the active pipeline
			Pipeline* getActivePipeline();

			//! Returns the active pipeline name
			std::string getActivePipelineName();
			//! Sets the named pipeline as the active pipeline for rendering purposes
			void setActivePipeline (const std::string &pipelineName);
			//! Returns the number of pipelines
			unsigned int getNumPipelines();
			//! Returns a vector with the name of all the defined pipelines
			std::vector<std::string> *getPipelineNames();

			// PASSES
			//! Checks if a given named pass exists in the named pipeline
			bool hasPass(const std::string &pipeline, const std::string &pass);
			//! Returns the named pass. Returns NULL if the pass does not exist
			Pass *getPass(const std::string &pipeline, const std::string &pass);
			//! Returns the named pass of the current pipeline. Returns NULL if the pass does not exist
			Pass *getPass(const std::string &passName);
			//! Returns the current pass. NULL if there is none
			Pass *getCurrentPass();

			void *getCurrentPassAttribute(std::string paramName, Enums::DataType dt);
			void *getPassAttribute(std::string passName, std::string paramName, Enums::DataType dt);
			Enums::DataType getPassAttributeType(std::string paramName);

			//! Returns the name of the last pass' camera from the active pipeline
			const std::string &getDefaultCameraName();

			//void reload (void);
			//void sendKeyToEngine (char keyCode); 

			//! Currently does nothing. Returns -1
			int pick (int x, int y, std::vector<nau::scene::SceneObject*> &objects, nau::scene::Camera &aCamera);
		
			// TRIANGLE IDS
			void prepareTriangleIDs(bool ids);
			void getVertexData(unsigned int sceneObjID, 
									 unsigned int triID);
			SceneObject *getSceneObject(int id);
			void addSceneObject(SceneObject *s);
			void deleteSceneObject(int id);

			// RENDER QUEUE
			//! Clear Renderqueue
			void clearQueue (void);
			//! Add an ISceneObject to the IRenderQueue
			void addToQueue (nau::scene::SceneObject *aObject, 
				std::map<std::string, nau::material::MaterialID> &materialMap);
			//! Calls the IRenderQueue processQueue method
			void processQueue (void);
			

			// CAMERAS
			//! Checks if the given named camera exists
			bool hasCamera (const std::string &cameraName);
			//! Returns a pointer to the given named camera
			nau::scene::Camera* getCamera (const std::string &cameraName);
			//! Returns the number of cameras
			unsigned int getNumCameras();
			//! Returns a vector with the name of all cameras
			std::vector<std::string> *getCameraNames();
			//! Returns the camera of the pass currently in execution. if no pass is being rendered it returns the pipeline's default camera
			nau::scene::Camera* getCurrentCamera();

			// VIEWPORTS
			//! Calls the Renderer to set the viewport
			void setViewport(nau::render::Viewport *vp);

			// LIGHTS
			//! Checks to see if the given named light exists
			bool hasLight (const std::string &lightName);
			//! Returns a pointer to the given named light. If the light does not exist it creates one
			nau::scene::Light* getLight (const std::string &lightName);
			//! Returns the named light. If it does not exist, it creates a light of the given class
			nau::scene::Light* getLight (const std::string &lightName, const std::string &lightClass);
			//! Returns the number of lights
			unsigned int getNumLights();
			//! Returns a vector with the name of all the lights
			std::vector<std::string> *getLightNames();

			// SCENES
			bool hasScene (const std::string &sceneName);
			nau::scene::IScene* createScene (const std::string &sceneName, const std::string &sceneType = "OctreeUnified");
			//! Return the named scene. If it does not exist it creates one
			nau::scene::IScene* getScene (const std::string &sceneName);
			//! Returns all the scene names, but the SceneAux type
			std::vector<std::string> *getSceneNames();
			//! Returns ALL the scene names
			std::vector<std::string> *getAllSceneNames();

			// MATERIALS
			//! Returns all the material names from the loaded scenes
			void materialNamesFromLoadedScenes (std::vector<std::string> &materials);

			//void addAlgorithm (std::string algorithm); /***MARK***/ //This shouldn't create the algorithm
			// But then again, this will, probably, not be like this
			
			//! Sets the RenderMode: Wireframe, point, solid or material
			void setRenderMode (nau::render::IRenderer::TRenderMode mode);

		public:
			//! Constructor
			RenderManager(void);

		public:
			//! Destructor. Calls clear to delete the data
			~RenderManager(void);
		};
	};
};
#endif // RENDERMANAGER_H
