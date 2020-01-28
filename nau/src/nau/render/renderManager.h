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
			//std::vector<SceneObject*> m_SceneObjects;

			std::map<std::string, std::shared_ptr<nau::scene::IScene>> m_Scenes;
			std::map<std::string, std::shared_ptr<nau::scene::Camera>> m_Cameras;
			std::map<std::string, std::shared_ptr<nau::scene::Light>> m_Lights;
			std::map<std::string, std::shared_ptr<Viewport>> m_Viewports;

			unsigned int m_ActivePipelineIndex;
			
			typedef enum {
				RUN_DEFAULT,
				RUN_ALL
			} RunMode;

			RunMode m_RunMode;
			std::string m_DefaultCamera;

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
			nau_API std::shared_ptr<Viewport> &createViewport(const std::string &name, nau::math::vec4 &bgColor);
			nau_API std::shared_ptr<Viewport> &createViewport(const std::string &name);
			nau_API std::shared_ptr<Viewport> getViewport(const std::string &name);
			nau_API void getViewportNames(std::vector<std::string> *);
			nau_API bool hasViewport(const std::string &name);


			// PIPELINES
			nau_API std::shared_ptr<Pipeline> &createPipeline(const std::string &pipelineName);
			//! Checks if a given named pipeline exists
			nau_API bool hasPipeline (const std::string &pipelineName);
			//! Returns a pointer to the named pipeline
			nau_API std::shared_ptr<Pipeline> &getPipeline (const std::string &pipelineName);
			nau_API unsigned int getPipelineIndex (const std::string &pipelineName);
			//! Returns a pointer to the active pipeline
			nau_API std::shared_ptr<Pipeline> &getActivePipeline();

			//! Returns the active pipeline name
			nau_API std::string getActivePipelineName();
			nau_API int getActivePipelineIndex();

			//! Sets the named pipeline as the active pipeline for rendering purposes
			nau_API void setActivePipeline (const std::string &pipelineName);
			nau_API void setActivePipeline(int index);
			//! Sets the named pipeline as the active pipeline for rendering purposes
			nau_API void setActivePipeline (unsigned int index);
			//! Returns the number of pipelines
			nau_API unsigned int getNumPipelines();
			//! Returns a vector with the name of all the defined pipelines
			nau_API void getPipelineNames(std::vector<std::string> *);

			bool setRunMode(std::string s);

			// PASSES
			//! Checks if a given named pass exists in the named pipeline
			nau_API bool hasPass(const std::string &pipeline, const std::string &pass);
			//! Returns the named pass. Returns NULL if the pass does not exist
			nau_API Pass *getPass(const std::string &pipeline, const std::string &pass);
			//! Returns the named pass of the current pipeline. Returns NULL if the pass does not exist
			nau_API Pass *getPass(const std::string &passName);
			//! Returns the current pass. NULL if there is none
			nau_API Pass *getCurrentPass();

			nau_API void *getCurrentPassAttribute(std::string paramName, Enums::DataType dt);
			nau_API void *getPassAttribute(std::string passName, std::string paramName, Enums::DataType dt);
			nau_API Enums::DataType getPassAttributeType(std::string paramName);

			//! Returns the name of the last pass' camera from the active pipeline
			nau_API const std::string &getDefaultCameraName();

			//! Currently does nothing. Returns -1
			int pick (int x, int y, std::vector<std::shared_ptr<SceneObject>> &objects, nau::scene::Camera &aCamera);
		
			// TRIANGLE IDS
			void prepareTriangleIDs(bool ids);
			//void getVertexData(unsigned int sceneObjID, 
			//						 unsigned int triID);
			//SceneObject *getSceneObject(int id);
			//void addSceneObject(SceneObject *s);
			//void deleteSceneObject(int id);

			// RENDER QUEUE
			//! Clear Renderqueue
			nau_API void clearQueue (void);
			//! Add an ISceneObject to the IRenderQueue
			nau_API void addToQueue (std::shared_ptr<SceneObject> &aObject,
				std::map<std::string, nau::material::MaterialID> &materialMap);
			//! Calls the IRenderQueue processQueue method
			nau_API void processQueue (void);
			

			// CAMERAS
			//! Checks if the given named camera exists
			nau_API bool hasCamera (const std::string &cameraName);
			//! Returns a pointer to the given named camera
			nau_API std::shared_ptr<nau::scene::Camera> &getCamera (const std::string &cameraName);
			//! Returns the number of cameras
			nau_API unsigned int getNumCameras();
			//! Returns a vector with the name of all cameras
			nau_API void getCameraNames(std::vector<std::string> *);
			//! Returns the camera of the pass currently in execution. if no pass is being rendered it returns the pipeline's default camera
			nau_API std::shared_ptr<nau::scene::Camera> &getCurrentCamera();
			nau_API std::shared_ptr<Camera> &createCamera(const std::string &name);

			// LIGHTS
			//! Checks to see if the given named light exists
			nau_API bool hasLight (const std::string &lightName);
			//! Returns a pointer to the given named light. If the light does not exist it creates one
			nau_API std::shared_ptr<nau::scene::Light> &getLight (const std::string &lightName);
			//! Returns the named light. If it does not exist, it creates a light of the given class
			nau_API std::shared_ptr<nau::scene::Light> &createLight (const std::string &lightName, const std::string &lightClass="default");
			//! Returns the number of lights
			nau_API unsigned int getNumLights();
			//! Returns a vector with the name of all the lights
			nau_API void getLightNames(std::vector<std::string> *);

			// SCENES
			nau_API bool hasScene (const std::string &sceneName);
			nau_API std::shared_ptr<nau::scene::IScene> & createScene (const std::string &sceneName, const std::string &sceneType = "OctreeUnified");
			//! Return the named scene. If it does not exist it creates one
			nau_API std::shared_ptr<nau::scene::IScene> & getScene (const std::string &sceneName);
			//! Returns all the scene names, but the SceneAux type
			nau_API void getSceneNames(std::vector<std::string> *);
			//! Returns ALL the scene names
			nau_API void getAllSceneNames(std::vector<std::string> *);
			// OCTREE STUFF
			//! Creates an octree for every OctreeScene
			nau_API void buildOctrees();

			//! Create VBOs for every IScene, erases all vertex data, except vertex coordinates 
			void compile();

			// MATERIALS
			//! Returns all the material names from the loaded scenes
			void materialNamesFromLoadedScenes (std::vector<std::string> &materials);

			//void addAlgorithm (std::string algorithm); /***MARK***/ //This shouldn't create the algorithm
			// But then again, this will, probably, not be like this
			
			//! Sets the RenderMode: Wireframe, point, solid or material
			nau_API void setRenderMode (nau::render::IRenderer::TRenderMode mode);
			nau_API void resetRenderMode();
			nau_API void applyRenderMode();

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
