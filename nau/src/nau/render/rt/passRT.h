#ifndef NAU_RT_PASS_H
#define NAU_RT_PASS_H

#include "nau/config.h"
#if NAU_RT == 1




#include <vector>
#include <string>
#include <map>



#include "nau/render/pass.h"
#include "nau/render/rt/rtBuffer.h"
#include "nau/render/rt/rtProgramManager.h"
#include "nau/render/rt/rtRenderer.h"

#include <glbinding/gl/gl.h>
#include <cuda_gl_interop.h>
#include <cuda.h>


namespace nau
{
	namespace render
	{
		namespace rt {

			class PassRT : public Pass {

				friend class PassFactory;

			public:

				struct LaunchParams
				{
					int       frame;
					uint32_t* colorBuffer;
					int2     fbSize;
				} launchParams;



				virtual ~PassRT();

				// creates pass from PassFactory class
				static std::shared_ptr<Pass> Create(const std::string& name);

				virtual void addScene(const std::string& sceneName);
				virtual void setRenderTarget(nau::render::IRenderTarget* rt);

				virtual void prepare(void);
				virtual void restore(void);
				virtual void doPass(void);

				/*! add a ray type. the index is incremental*/
				void addRayType(const std::string& name);
				// add a vertex attribute
				void addVertexAttribute(unsigned int attr);

				void setRayGenProcedure(const std::string &file, const std::string &proc);
				void setDefaultProc(const std::string& pRayType, int procType, const std::string& pFile, const std::string& pName);

			protected:
				PassRT(const std::string& passName);
				static bool Init();
				static bool Inited;

				void rtInit();


				bool m_RTisReady;
				bool m_RThasIssues;

				RTProgramManager m_ProgramManager;


				// stores a boolean for each vertex attribute provided 
				// in vertexData class
				std::vector<bool> m_VertexAttributes;

				// ray tracing outputs
				std::vector<RTBuffer> m_OutputOptix;
				std::vector<unsigned int> m_OutputPBO;
				std::vector<cudaGraphicsResource *> m_OutputCGR;
				std::vector<unsigned int> m_OutputTexIDs;
				std::vector<unsigned char*> m_OutputBufferPrs;

			};
		};
	};
};

#endif 

#endif




