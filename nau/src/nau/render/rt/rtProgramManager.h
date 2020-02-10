#ifndef NAU_RT_PROGRAM_MANAGER_H
#define NAU_RT_PROGRAM_MANAGER_H

#include "nau/config.h"

#if NAU_RT == 1

#include <map>
#include <string>
#include <vector>

#include "optix.h"
#include "optix_types.h"

#include "nau/material/iTexture.h"
#include "nau/render/rt/rtBuffer.h"
#include "nau/render/rt/rtGeometry.h"


namespace nau {
	namespace render {
		namespace rt {


			class RTProgramManager {

			public:

				RTProgramManager();
				~RTProgramManager();

				struct ProgramInfo {
					OptixProgramGroup hitProgram;
					unsigned int RayType;
					std::string moduleNameCH;
					std::string programNameCH;
					std::string moduleNameAH;
					std::string programNameAH;

					OptixProgramGroup missProgram;
					std::string moduleNameMiss;
					std::string programNameMiss;
				};

				const std::vector<std::string>& getPtxFiles();
				const std::map<std::string, std::map<int, ProgramInfo>>& getProgramInfo();

				const OptixPipeline& getPipeline();
				const OptixShaderBindingTable& getSBT();

				bool generatePrograms();
				bool generateModules();
				bool generatePipeline();
				bool generateSBT(const std::map<std::string, RTGeometry::CUDABuffers> &cuBuffers);

				void setRayGenProcedure(const std::string &file, const std::string &proc);
				void setDefaultProc(const std::string& pRayType, int procType, const std::string& pFile, const std::string& pName);
				void addRayType(const std::string& name);

				// create cuda textures from OpenGL textures
				bool processTextures();

			protected:
				
				bool typeIsOK(nau::material::ITexture* t);
				
				// vector of ptx files
				std::vector<std::string> m_PtxFiles;


				// stores the names of the ray types. Indices are incremental
				std::map<std::string, unsigned int> m_RayTypes;

				std::string m_RayGenProcName;
				std::string m_RayGenFile;
				OptixProgramGroup m_RayGenProgramGroup;

				// material name -> (ray type -> ProgramInfo)
				std::map<std::string, std::map<int, ProgramInfo>> m_Materials;

				// proc name -> module
				std::map<std::string, OptixModule>  m_Module;
				OptixModuleCompileOptions           m_ModuleCompileOptions;

				OptixPipeline               m_Pipeline;
				OptixPipelineCompileOptions m_PipelineCompileOptions;
				OptixPipelineLinkOptions    m_PipelineLinkOptions;

				// -------------------------------------------------------------------------
				// SBT data
				template <typename T>
				struct SbtRecord
				{
					__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
					T data;
				};

				struct RayGenData {
					int3 color;
				};

				struct MissData {
					void* data;
				};

				struct vertexData {
					float4* position;
					float4* normal;
					float4* texCoord0;
					float4* tangent;
					float4* bitangent;
				};

				struct HitGroupData {
					uint3* index;
					vertexData vertexD;
					int hasTexture;
					cudaTextureObject_t texture;
					float3 color;
				};

				typedef SbtRecord<RayGenData>     RaygenRecord;
				typedef SbtRecord<MissData>       MissRecord;
				typedef SbtRecord<HitGroupData>   HitgroupRecord;

				OptixShaderBindingTable m_SBT = {};


				struct TextureData {
					cudaGraphicsResource* cgr;
					cudaTextureObject_t cto;
					cudaArray *ca;
				};

				std::map<int, TextureData > m_Textures;
			};
		};
	};
};

#endif
#endif