#include "nau/config.h"

#if NAU_RT == 1

#include "nau/render/rt/rtProgramManager.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/material/iTexture.h"
#include "nau/resource/resourceManager.h"
#include "nau/render/rt/rtException.h"
#include "nau/render/rt/rtRenderer.h"
#include "nau/system/file.h"

#include <assert.h>

using namespace nau::render::rt;


RTProgramManager::RTProgramManager() {

}


RTProgramManager::~RTProgramManager() {

}


const std::map<std::string, std::map<int, RTProgramManager::ProgramInfo>>&
RTProgramManager::getProgramInfo() {

	return m_Materials;
}



const OptixPipeline& 
RTProgramManager::getPipeline() {

	return m_Pipeline;
}


const OptixShaderBindingTable& 
RTProgramManager::getSBT()
{
	return m_SBT;
}


bool
RTProgramManager::typeIsOK(ITexture* t) {

	if (t->getPrope(ITexture::DIMENSION) != GL_TEXTURE_2D)
		return false;

	int format = t->getPrope(ITexture::FORMAT);

	if (format == GL_DEPTH_COMPONENT || format == gl::GL_DEPTH_STENCIL)
		return false;

	return true;
}


bool
RTProgramManager::processTextures() {

	int texCount = RESOURCEMANAGER->getNumTextures();

	for (int i = 0; i < texCount; ++i) {

		ITexture *t = RESOURCEMANAGER->getTexture(i);

		// initially offer support for 2D textures only
		if (t->getPrope(ITexture::DIMENSION) == GL_TEXTURE_2D && typeIsOK(t)) {

			try {
				int id = t->getPropi(ITexture::ID);
				m_Textures[id] = {};

				CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_Textures[id].cgr, id,
					GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
				CUDA_CHECK(cudaGraphicsMapResources(1, &m_Textures[id].cgr, 0));
				CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&m_Textures[id].ca, m_Textures[id].cgr, 0, 0));

				cudaResourceDesc resDesc;
				memset(&resDesc, 0, sizeof(resDesc));
				resDesc.resType = cudaResourceTypeArray;
				resDesc.res.array.array = m_Textures[id].ca;

				cudaTextureDesc texDesc;
				memset(&texDesc, 0, sizeof(texDesc));
				texDesc.addressMode[0] = cudaAddressModeWrap;
				texDesc.addressMode[1] = cudaAddressModeWrap;
				texDesc.filterMode = cudaFilterModePoint;
				texDesc.readMode = cudaReadModeNormalizedFloat;
				texDesc.normalizedCoords = 1;
				texDesc.maxAnisotropy = 1;
				texDesc.maxMipmapLevelClamp = 99;
				texDesc.minMipmapLevelClamp = 0;
				texDesc.mipmapFilterMode = cudaFilterModePoint;
				texDesc.borderColor[0] = 1.0f;
				texDesc.sRGB = 0;
				CUDA_CHECK(cudaCreateTextureObject(&m_Textures[id].cto, &resDesc, &texDesc, nullptr));
			}
			catch (std::exception const& e) {
				SLOG("Exception generating texture: %s", e.what());
				SLOG("Exception in texture: %s", t->getLabel().c_str());
			}
		}
	}
	return true;
}


bool
RTProgramManager::generateModules()
{
	char log[2048];
	size_t sizeof_log = sizeof(log);
	try {
		m_ModuleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		m_ModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		m_ModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

		m_PipelineCompileOptions = {};
		m_PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		m_PipelineCompileOptions.usesMotionBlur = false;
		m_PipelineCompileOptions.numPayloadValues = 2;
		m_PipelineCompileOptions.numAttributeValues = 2;
		m_PipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH  | OPTIX_EXCEPTION_FLAG_DEBUG; // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
		m_PipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

		m_PipelineLinkOptions.overrideUsesMotionBlur = false;
		m_PipelineLinkOptions.maxTraceDepth = 10;


		const std::vector<std::string>& ptxFiles = getPtxFiles();

		for (size_t i = 0; i < ptxFiles.size(); ++i) {

			std::string ptxCode = nau::system::File::TextRead(ptxFiles[i]);

			sizeof_log = 2048;
			OPTIX_CHECK(optixModuleCreateFromPTX(RTRenderer::getOptixContext(),
				&m_ModuleCompileOptions,
				&m_PipelineCompileOptions,
				ptxCode.c_str(),
				ptxCode.size(),
				log, &sizeof_log,
				&m_Module[ptxFiles[i]]
			));
			if (sizeof_log > 1) 
				SLOG("RT: create modules - %s", log);
		}
	}
	catch (std::exception const& e) {
		SLOG("Exception generate modules: %s", e.what());
		if (sizeof_log > 1)
			SLOG("Exception in generate modules: %s", log);
		return false;
	}
	return true;
}


bool 
RTProgramManager::generatePrograms() {


	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};


	try {
		// add raygen program
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		pgDesc.raygen.module = m_Module[m_RayGenFile];
		pgDesc.raygen.entryFunctionName = m_RayGenProcName.c_str();
		OPTIX_CHECK(optixProgramGroupCreate(RTRenderer::getOptixContext(),
			&pgDesc,
			1,
			&pgOptions,
			log, &sizeof_log,
			&m_RayGenProgramGroup
		));
		if (sizeof_log > 1) SLOG("RT: create ray gen program - %s", log);

		// for each material
		// mat is a pair (material name -> (rayType -> programInfo))
		for (auto &mat : m_Materials) {

			// for each ray type
			// proc is a pair (rayType -> programInfo)
			for (auto &proc : mat.second) {

				OptixProgramGroupDesc pgDescHit = {};
				pgDescHit.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				pgDescHit.hitgroup.moduleAH = m_Module[proc.second.moduleNameAH];
				pgDescHit.hitgroup.entryFunctionNameAH = proc.second.programNameAH.c_str();
				pgDescHit.hitgroup.moduleCH = m_Module[proc.second.moduleNameCH];
				pgDescHit.hitgroup.entryFunctionNameCH = proc.second.programNameCH.c_str();

				sizeof_log = 2048;
				OPTIX_CHECK(optixProgramGroupCreate(RTRenderer::getOptixContext(),
					&pgDescHit,
					1,
					&pgOptions,
					log, &sizeof_log,
					&(proc.second.hitProgram)
				));
				if (sizeof_log > 1) 
					SLOG("RT: create hit group for material %s - %s", mat.first.c_str(), log);

				OptixProgramGroupDesc pgDescMiss = {};
				pgDescMiss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
				pgDescMiss.miss.module = m_Module[proc.second.moduleNameMiss];
				pgDescMiss.miss.entryFunctionName = proc.second.programNameMiss.c_str();

				sizeof_log = 2048;
				OPTIX_CHECK(optixProgramGroupCreate(RTRenderer::getOptixContext(),
					&pgDescMiss,
					1,
					&pgOptions,
					log, &sizeof_log,
					&(proc.second.missProgram)
				));
				if (sizeof_log > 1) 
					SLOG("RT: create miss group for material %s - %s", mat.first.c_str(), log);
				
			}
		}
	}
	catch (std::exception const& e)	{
		SLOG("Exception in generate programs: %s", e.what());
		if (sizeof_log > 1)
			SLOG("Exception in generate programs: %s", log);
		return false;
	}
	return true;
}


bool
RTProgramManager::generatePipeline() {

	char log[2048];
	size_t sizeof_log = sizeof(log);

	try {

		std::vector<OptixProgramGroup> programGroups;

		// fill in programGroups
		for (auto &mat : m_Materials) {
			for (auto &rayTypeInfo : mat.second) {
				programGroups.push_back(rayTypeInfo.second.hitProgram);
				programGroups.push_back(rayTypeInfo.second.missProgram);

			}
		}

		programGroups.push_back(m_RayGenProgramGroup);

		OPTIX_CHECK(optixPipelineCreate(RTRenderer::getOptixContext(),
			&m_PipelineCompileOptions,
			&m_PipelineLinkOptions,
			programGroups.data(),
			(int)programGroups.size(),
			log, &sizeof_log,
			&m_Pipeline
		));
		if (sizeof_log > 1)
			SLOG("RT: create pipeline: %s", log);

		OPTIX_CHECK(optixPipelineSetStackSize(
			m_Pipeline, 2 * 1024, 2 * 1024, 2 * 1024, 1));
	} 
	catch (std::exception const& e) {
		SLOG("Exception in generate pipeline: %s", e.what());
		if (sizeof_log > 1)
			SLOG("Exception in generate pipeline: %s", log);
		return false;
	}
	return true;
}


/*! constructs the shader binding table */
bool 
RTProgramManager::generateSBT(const std::map<std::string, RTGeometry::CUDABuffers> &cuBuffers) {

	std::vector<MissRecord> missRecords;
	std::vector<HitgroupRecord> hitgroupRecords;

	try {
		// build ray gen record
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(m_RayGenProgramGroup, &rec));
		rec.data.color = make_int3(255, 0, 0);

		RTBuffer buff;
		buff.store((void*)&rec, sizeof(RaygenRecord));
		m_SBT.raygenRecord = buff.getPtr();


		// build hit and miss records
		for (auto &scObj: cuBuffers) {

			const std::map<int, RTGeometry::CUDABuffer> &vertexB = scObj.second.vertexBuffers;
			const std::map<std::string, RTGeometry::CUDABuffer> &indexB = scObj.second.indexBuffers;

			std::string mat;
			float3 color;
			for (auto &indexBuffer:indexB) {
				
				std::vector<unsigned int> textureIDs;
				std::shared_ptr<Material> &nauMaterial = MATERIALLIBMANAGER->getMaterialFromDefaultLib(indexBuffer.first);
				nauMaterial->getTextureIDs(&textureIDs);

				// if material is not in the material map use default mat
				if (m_Materials.count(indexBuffer.first) == 0) {
					mat = "default";
				}
				else {
					mat = indexBuffer.first;
				}
				vec4 x = nauMaterial->getColor().getPropf4(ColorMaterial::DIFFUSE);;
				color = make_float3(x.x, x.y, x.z);
				
				std::map<int, ProgramInfo>& material = m_Materials[mat];


				for (int rayType = 0; rayType < material.size(); rayType++) {

					ProgramInfo& pi = material[rayType];

					MissRecord recM;
					OPTIX_CHECK(optixSbtRecordPackHeader(pi.missProgram, &recM));
					recM.data.data = nullptr;
					missRecords.push_back(recM);


					HitgroupRecord recH;
					OPTIX_CHECK(optixSbtRecordPackHeader(pi.hitProgram, &recH));
					recH.data.color = color;
					recH.data.vertexD.position = (float4 *)(vertexB.at(0).memPtr);
					if (vertexB.size() > 1)
						recH.data.vertexD.normal = (float4 *)vertexB.at(1).memPtr;
					if (vertexB.size() > 2)
						recH.data.vertexD.texCoord0 = (float4*)vertexB.at(2).memPtr;
					if (vertexB.size() > 3)
						recH.data.vertexD.tangent = (float4*)vertexB.at(3).memPtr;
					if (vertexB.size() > 4)
						recH.data.vertexD.bitangent = (float4*)vertexB.at(4).memPtr;

					if (textureIDs.size() != 0) {
						recH.data.hasTexture = 1;
						recH.data.texture = m_Textures[textureIDs[0]].cto;
					}
					else
						recH.data.hasTexture = 0;
					recH.data.index = (uint3 *)indexBuffer.second.memPtr;
					hitgroupRecords.push_back(recH);

				}
			}

		}

		RTBuffer missRecordsBuffer;
		missRecordsBuffer.store((void*)&missRecords[0], sizeof(MissRecord) * missRecords.size());
		m_SBT.missRecordBase = missRecordsBuffer.getPtr();
		m_SBT.missRecordStrideInBytes = sizeof(MissRecord);
		m_SBT.missRecordCount = (int)missRecords.size();

		RTBuffer hitgroupRecordsBuffer;
		hitgroupRecordsBuffer.store((void*)&hitgroupRecords[0], sizeof(HitgroupRecord) * hitgroupRecords.size());
		m_SBT.hitgroupRecordBase = hitgroupRecordsBuffer.getPtr();
		m_SBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		m_SBT.hitgroupRecordCount = (int)hitgroupRecords.size();

	}
	catch (std::exception const& e) {
		SLOG("Exception in generate SBT: %s", e.what());
		return false;
	}
	return true;
}


const std::vector<std::string>& 
RTProgramManager::getPtxFiles() {

	return m_PtxFiles;
}


void
RTProgramManager::addRayType(const std::string& name) {

	if (!m_RayTypes.count(name))
		m_RayTypes[name] = (int)m_RayTypes.size();
}


void
RTProgramManager::setRayGenProcedure(const std::string &file, const std::string &proc) {

	m_PtxFiles.push_back(file);
	m_RayGenFile = file;
	m_RayGenProcName = proc;
}

void 
RTProgramManager::setDefaultProc(const std::string& pRayType, int procType, const std::string& pFile, const std::string& pName) {

	int rayTypeIndex = m_RayTypes[pRayType];

	assert(rayTypeIndex != -1);

	switch (procType) {
	case RTRenderer::ANY_HIT: 
		m_Materials["default"][rayTypeIndex].moduleNameAH = pFile;
		m_Materials["default"][rayTypeIndex].programNameAH = pName;
		break;
	case RTRenderer::CLOSEST_HIT:
		m_Materials["default"][rayTypeIndex].moduleNameCH = pFile;
		m_Materials["default"][rayTypeIndex].programNameCH = pName;
		break;
	case RTRenderer::MISS:
		m_Materials["default"][rayTypeIndex].moduleNameMiss = pFile;
		m_Materials["default"][rayTypeIndex].programNameMiss = pName;
		break;

	}

	bool ptxFound = false;
	for (int i = 0; !ptxFound && i < m_PtxFiles.size(); ++i)
		if (m_PtxFiles[i] == pFile)
			ptxFound = true;

	if (!ptxFound)
		m_PtxFiles.push_back(pFile);
}



#endif