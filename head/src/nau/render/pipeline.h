#ifndef PIPELINE_H
#define PIPELINE_H

#include <deque>
#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include <nau/scene/camera.h>
#include <nau/scene/sceneobject.h>
#include <nau/render/pass.h>
#include <nau/render/istate.h>

#define PIPE_PASS_MIDDLE 0
#define PIPE_PASS_START 1
#define PIPE_PASS_STARTEND 2
#define PIPE_PASS_END 3

namespace nau
{
	namespace render
	{

		class Pipeline {

		public:
			static const int MAXPASSEs = 16;
		private:

			std::deque<Pass *> m_Passes;
			std::string m_Name;

			//! The default camera is the camera that, by default, will receive events from the EventManager
			std::string m_DefaultCamera;

			Pass *m_CurrentPass;

			bool m_Active;

			int m_NextPass;


		public:
			Pipeline (std::string pipelineName = "Default");
			
			const std::string &getLastPassCameraName(); 
			int getNumberOfPasses();
			std::vector<std::string> *getPassNames();

			/** 
			 * Add a pass to the pipeline. 
			 * 
			 * \param aPass The pass object to add
			 * \param PassIndex The pipeline position to insert the pass. 
			 *                  0 is the first pass. -1 is the last. 
			 */
			void addPass (Pass* aPass, int PassIndex = -1);
			Pass* createPass (const std::string &name, const std::string &passName = "default");

			bool hasPass(const std::string &passName);
			Pass* getPass (const std::string &passName);
			Pass* getPass (int n);

			//! Gets the name of the camera from the current pass being executed
			const std::string &getCurrentCamera();

			//! Gets the default camera, if not set it returns the last pass camera
			const std::string &getDefaultCameraName();
			//! Sets the default camera name. No error checking is performed!
			void setDefaultCamera(const std::string &defCam);

			void initState(IState *state);

			Pass *getCurrentPass();

			unsigned char execute (/*nau::scene::Camera* aCamera, nau::scene::IScene *aScene*/);
		
			bool isActive (void);

		  
		protected:
			  Pipeline (const Pipeline&);
			  Pipeline& operator= (const Pipeline&);

		//private:
			//float lightPos[8][4],spotDir[8][3];
			//int lightWhere[8];
			//TrackBall tb;

			//CCamera *m_defaultCamera;
			//int m_activeCameraIndex;
			//CProjection *m_defaultProjection;
			//CViewport *m_defaultViewport;
			
			//int vpw,vph; // window viewport dimension

			//CMaterialLibManager *m_materialLibManager;
			//std::vector<CMaterial *> m_mats;

			//void setCamera(	CCamera *cam,CProjection *proj,CViewport *vp);
			//int getActiveCameraIndex();
			//void setActiveCameraIndex(int i);
			//void setPassCamera(int pass, CCamera *cam);			
			//std::string m_Path;
			//std::string m_Filename;
			
			//IState *m_GlCurrState, *m_GlDefState;

			//std::string m_DefaultCamera;

		};
	};
};

#endif

