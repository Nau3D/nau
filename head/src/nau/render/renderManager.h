#ifndef RENDERMANAGER_H
#define RENDERMANAGER_H


#include "nau/render/pipeline.h"
#include "nau/render/iRenderer.h"
#include "nau/render/iRenderQueue.h"
#include "nau/render/viewport.h"
#include "nau/scene/camera.h"
#include "nau/scene/iScene.h"
#include "nau/scene/light.h"
#include "nau/scene/sceneObject.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

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
			std::unique_ptr<IRenderer> m_pRenderer;
			std::unique_ptr<IRenderQueue> m_pRenderQueue;
			std::vector<std::shared_ptr<Pipeline>> m_Pipelines;
			std::vector<SceneObject*> m_SceneObjects;

			std::map<std::string, std::shared_ptr<IScene>> m_Scenes;
			std::map<std::string, std::shared_ptr<Camera>> m_Cameras;
			std::map<std::string, std::shared_ptr<Light>> m_Lights;
			std::map<std::string, std::shared_ptr<Viewport>> m_Viewports;

			unsigned int m_ActivePipelineIndex;
			
			typedef enum {
				RUN_DEFAULT,
				RUN_ALL
			} RunMode;

			RunMode m_RunMode;

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

			
			// VIEWPORTS
			std::shared_ptr<Viewport> createViewport(const std::string &name, nau::math::vec4 &bgColor);
			std::shared_ptr<Viewport> createViewport(const std::string &name);
			std::shared_ptr<Viewport> getViewport(const std::string &name);
			void getViewportNames(std::vector<std::string> *);
			bool hasViewport(const std::string &name);


			// PIPELINES
			std::shared_ptr<Pipeline> &createPipeline(const std::string &pipelineName);
			//! Checks if a given named pipeline exists
			bool hasPipeline (const std::string &pipelineName);
			//! Returns a pointer to the named pipeline
			std::shared_ptr<Pipeline> &getPipeline (const std::string &pipelineName);
			unsigned int getPipelineIndex (const std::string &pipelineName);
			//! Returns a pointer to the active pipeline
			std::shared_ptr<Pipeline> &getActivePipeline();

			//! Returns the active pipeline name
			std::string getActivePipelineName();
			//! Sets the named pipeline as the active pipeline for rendering purposes
			void setActivePipeline (const std::string &pipelineName);
			//! Sets the named pipeline as the active pipeline for rendering purposes
			void setActivePipeline (unsigned int index);
			//! Returns the number of pipelines
			unsigned int getNumPipelines();
			//! Returns a vector with the name of all the defined pipelines
			void getPipelineNames(std::vector<std::string> *);

			bool setRunMode(std::string s);

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
			std::shared_ptr<Camera> &getCamera (const std::string &cameraName);
			//! Returns the number of cameras
			unsigned int getNumCameras();
			//! Returns a vector with the name of all cameras
			void getCameraNames(std::vector<std::string> *);
			//! Returns the camera of the pass currently in execution. if no pass is being rendered it returns the pipeline's default camera
			std::shared_ptr<Camera> &getCurrentCamera();


			// LIGHTS
			//! Checks to see if the given named light exists
			bool hasLight (const std::string &lightName);
			//! Returns a pointer to the given named light. If the light does not exist it creates one
			std::shared_ptr<Light> &getLight (const std::string &lightName);
			//! Returns the named light. If it does not exist, it creates a light of the given class
			std::shared_ptr<Light> &createLight (const std::string &lightName, const std::string &lightClass);
			//! Returns the number of lights
			unsigned int getNumLights();
			//! Returns a vector with the name of all the lights
			void getLightNames(std::vector<std::string> *);

			// SCENES
			bool hasScene (const std::string &sceneName);
			std::shared_ptr<IScene> & createScene (const std::string &sceneName, const std::string &sceneType = "OctreeUnified");
			//! Return the named scene. If it does not exist it creates one
			std::shared_ptr<IScene> & getScene (const std::string &sceneName);
			//! Returns all the scene names, but the SceneAux type
			void getSceneNames(std::vector<std::string> *);
			//! Returns ALL the scene names
			void getAllSceneNames(std::vector<std::string> *);
			// OCTREE STUFF
			//! Creates an octree for every OctreeScene
			void buildOctrees();

			//! Create VBOs for every IScene, erases all vertex data, except vertex coordinates 
			void compile();

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
