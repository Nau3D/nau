#ifndef PIPELINE_H
#define PIPELINE_H


#include "nau/material/iState.h"
#include "nau/scene/camera.h"
#include "nau/scene/sceneObject.h"
#include "nau/render/pass.h"

#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>

#define PIPE_PASS_MIDDLE 0
#define PIPE_PASS_START 1
#define PIPE_PASS_STARTEND 2
#define PIPE_PASS_END 3

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

		class Pipeline {

			friend class RenderManager;
		public:

			~Pipeline();

			nau_API std::string getName();
			nau_API void getPassNames(std::vector<std::string> *);

			nau_API int getNumberOfPasses();
			nau_API int getPassCounter();

			/** 
			 * Add a pass to the pipeline. 
			 * 
			 * \param aPass The pass object to add
			 * \param PassIndex The pipeline position to insert the pass. 
			 *                  0 is the first pass. -1 is the last. 
			 */
			//void addPass (Pass* aPass, int PassIndex = -1);
			nau_API Pass* createPass (const std::string &name, const std::string &passName = "default");

			nau_API bool hasPass(const std::string &passName);
			nau_API Pass* getPass (const std::string &passName);
			nau_API Pass* getPass (int n);
			nau_API Pass *getCurrentPass();

			nau_API void setFrameCount(unsigned int k);
			nau_API unsigned int getFrameCount();

			//! Gets the name of the camera from the current pass being executed
			nau_API const std::string &getCurrentCamera();
			nau_API const std::string &getLastPassCameraName();

			//! Gets the default camera, if not set it returns the last pass camera
			nau_API const std::string &getDefaultCameraName();
			nau_API void setDefaultCamera(const std::string &defCam);

			nau_API void initState(IState *state);


			nau_API void execute();

			nau_API void executeNextPass();

		
			// -----------------------------------------------------------------
			//		PRE POST SCRIPTS
			// -----------------------------------------------------------------
			void setTestScript(const std::string &file, const std::string &name);
			void setPreScript(const std::string &file, const std::string &name);
			void setPostScript(const std::string &file, const std::string &name);
			void callPreScript();
			void callPostScript();
			bool callTestScript();


		protected:

			Pipeline(std::string pipelineName = "Default");

			void executePass(std::shared_ptr<Pass> &);
			Pipeline (const Pipeline&);
			Pipeline& operator= (const Pipeline&);

			std::deque<std::shared_ptr<Pass>> m_Passes;
			std::string m_Name;

			//! The default camera will receive events from the EventManager
			std::string m_DefaultCamera;

			std::shared_ptr<Pass> m_CurrentPass;

			unsigned int m_NextPass;

			std::string m_PreScriptFile, m_PreScriptName,
				m_PostScriptFile, m_PostScriptName,
				m_TestScriptFile, m_TestScriptName;

			unsigned int m_FrameCount = 0;
		};
	};
};

#endif

