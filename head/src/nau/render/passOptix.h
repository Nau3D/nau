#ifndef PASSOPTIX_H
#define PASSOPTIX_H

#include <vector>
#include <string>
#include <map>

#include "nau/render/optix/optixGeometry.h"
#include "nau/render/optix/optixmaterialLib.h"
#include "nau/render/optix/optixRenderer.h"
#include "nau/render/optix/optixTextureLib.h"

//#include <optix.h>
//#include <optixu/optixpp_namespace.h>
//#include <optixu/optixu_math_namespace.h>
using namespace nau::render::optixRender;

#include "nau/render/pass.h"

namespace nau
{
	namespace render
	{

		class PassOptix : public Pass {

		public:
			PassOptix (const std::string &passName);
			//Pass(std::string path,std::string fname);
			virtual ~PassOptix();

			virtual void addScene (const std::string &sceneName);
			virtual void setRenderTarget (nau::render::RenderTarget* rt);

			virtual void prepare (void);
			virtual void restore (void);
			virtual void doPass (void);

			void setOptixEntryPointProcedure(OptixRenderer::ProgramTypes type, 
										std::string pFile, 
										std::string pProc);
			void setDefaultMaterialProc(OptixMaterialLib::MaterialPrograms type, 
										std::string rayType,
										std::string pFile, 
										std::string pProc);
			void setMaterialProc(std::string name,
									OptixMaterialLib::MaterialPrograms type, 
									std::string rayType,
									std::string pFile, 
									std::string pProc);
			void setInputBuffer(std::string, std::string);
			void setOutputBuffer(std::string, std::string);

			void setGeometryIntersectProc(std::string, std::string);
			void setBoundingBoxProc(std::string, std::string);

			void addVertexAttribute(unsigned int attr);
			void addMaterialAttribute(std::string name, ProgramValue &p);
			void addGlobalAttribute(std::string name, ProgramValue &p);

		protected:
			virtual void setupCamera (void);
			RTformat getOptixFormat(Texture *texID);

			virtual void optixInit();
			// Optix Stuff
//			nau::render::optixRender::OptixRenderer o_Renderer;
			nau::render::optixRender::OptixBufferLib o_BufferLib;
			nau::render::optixRender::OptixGeometry o_GeomLib;
			nau::render::optixRender::OptixMaterialLib o_MatLib;
			nau::render::optixRender::OptixTextureLib o_TexLib;
			std::map<std::string, unsigned int> o_RayType;
			std::map<std::string, std::string> o_InputBuffers;
			unsigned int o_EntryPoint;
			optix::Context o_Context;
			bool o_OptixIsPrepared;

			std::vector<optix::Buffer> o_OutputBuffer;
			optix::GeometryGroup o_GeomGroup;
			std::vector<unsigned int> o_OutputPBO;
			optix::Material o_Material;
			std::vector<optix::GeometryInstance> o_GeomInstances;
			optix::Program o_GeometryIntersectionProgram, o_BoundingBoxProgram;
			optix::Program o_ClosestHitProgram;
			std::map<std::string, nau::material::ProgramValue> o_GlobalAttribute;

			// optix name > (gl tex name, PixelBufferID)
			typedef struct databuffer{
			
				std::string texName;
				unsigned int pbo;
			};
			std::map<std::string, databuffer> o_OutputDataBuffer;

		};
	};
};
#endif // PassOptix Class





	
	
