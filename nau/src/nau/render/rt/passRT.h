#ifndef NAU_RT_PASS_H
#define NAU_RT_PASS_H

#include "nau/config.h"
#if NAU_RT == 1

#include <vector>
#include <string>
#include <map>

#include "nau/render/pass.h"
#include "nau/render/rt/rtBuffer.h"
#include "nau/render/rt/rtGeometry.h"
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

				struct optixParams {
					char* data;
				};

				struct LaunchParams
				{
					struct {
						int frame;
						uint32_t* colorBuffer;
						int raysPerPixel;
					} frame;

					struct {
						float3 position;
						float3 direction;
						float3 horizontal;
						float3 vertical;
					} camera;

					OptixTraversableHandle traversable;

					optixParams *globalParams;

					LaunchParams() {
						frame.frame = 0;
						frame.colorBuffer = nullptr;
						frame.raysPerPixel = 1;

						camera.position = make_float3(0.0f, 0.0f, 0.0f);
						camera.direction = make_float3(0.0f, 0.0f, -1.0f);
						camera.horizontal = make_float3(1.0f, 0.0f, 0.0f);
						camera.vertical = make_float3(0.0f, 1.0f, 0.0f);

						globalParams = nullptr;
					}

				} launchParams;



				virtual ~PassRT();

				// creates pass from PassFactory class
				static std::shared_ptr<Pass> Create(const std::string& name);

				virtual void addScene(const std::string& sceneName);
				virtual void setRenderTarget(nau::render::IRenderTarget* rt);
				void bindCudaRenderTarget();
				void cleanCudaRenderTargetBindings();

				virtual void prepare(void);
				virtual void restore(void);
				virtual void doPass(void);

				virtual void setupCamera();

				/* add vertex to vertex attribute list */
				void addVertexAttribute(unsigned int  attr);

				/*! add a ray type. the index is incremental*/
				void addRayType(const std::string& name);

				void setRayGenProcedure(const std::string &file, const std::string &proc);
				void setDefaultProc(const std::string& pRayType, int procType, const std::string& pFile, const std::string& pName);
				void setMatProc(const std::string& matName, const std::string& pRayType, int procType, const std::string& pFile, const std::string& pName);

				void addParam(const std::string &name, const std::string &type, const std::string &context, const std::string &component, int id);
			protected:
				PassRT(const std::string& passName);
				static bool Init();
				static bool Inited;

				void rtInit();


				bool m_RTisReady;
				bool m_RThasIssues;

				RTProgramManager m_ProgramManager;
				RTGeometry m_Geometry;


				// stores a boolean for each vertex attribute provided 
				// in vertexData class
				std::vector<bool> m_VertexAttributes;

				// ray tracing outputs
				std::vector<RTBuffer> m_OutputOptix;
				std::vector<unsigned int> m_OutputPBO;
				std::vector<cudaGraphicsResource *> m_OutputCGR;
				std::vector<unsigned int> m_OutputTexIDs;
				std::vector<unsigned char*> m_OutputBufferPrs;

				RTBuffer m_LaunchParamsBuffer;

				uivec2 m_LaunchSize;

				struct Param{
					// param properties
					std::string name;
					std::string type;
					std::string context;
					std::string component;
					int id;
					// used to fill the memory block
					int offset;
					int size;
					Enums::DataType dt;
					int attr;
				} ;

				int computeParamsByteSize();
				void copyParamsToBuffer();
				std::vector<Param> m_Params;
				int m_ParamsSize;
				RTBuffer m_ParamsBuffer;


			};
		};
	};
};

#endif 

#endif




