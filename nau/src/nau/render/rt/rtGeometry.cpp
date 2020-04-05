#include "nau/config.h"
#if NAU_RT == 1

#include "nau/render/rt/rtGeometry.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/render/rt/rtBuffer.h"
#include "nau/render/rt/rtException.h"
#include "nau/render/rt/rtRenderer.h"
#include "nau/scene/iScene.h"
#include "nau/scene/sceneObject.h"


using namespace nau::render::rt;
using namespace gl;



RTGeometry::RTGeometry() {

	//m_VertexAttributes.resize(VertexData::MaxAttribs);
	rtTraversableHandle = 0;
}


void
RTGeometry::addVertexAttribute(unsigned int attr) {

	if (attr < VertexData::MaxAttribs)
		m_VertexAttributes.push_back(attr);
}


const OptixTraversableHandle& 
RTGeometry::getTraversableHandle() {
	return rtTraversableHandle;
}


const std::map<std::string, RTGeometry::CUDABuffers> & 
RTGeometry::getCudaVBOS() {
	return m_CudaVBOs;
}


bool 
RTGeometry::generateAccel(const std::vector<std::string>& sceneNames) {

	int meshCount = 0;

	if (sceneNames.size() == 0)
		return true;

	// for each scene create cuda buffers
	for (auto sceneName : sceneNames) {

		std::shared_ptr<IScene>& aScene = RENDERMANAGER->getScene(sceneName);
		aScene->compile();

		std::vector<std::shared_ptr<SceneObject>> sceneObjs;
		aScene->getAllObjects(&sceneObjs);

		for (auto sObj : sceneObjs) {
			std::shared_ptr<IRenderable> r = sObj->getRenderable();
			std::shared_ptr<VertexData>& v = r->getVertexData();

			// size in bytes
			size_t size = (size_t)v->getNumberOfVertices() * 4 * sizeof(float);
			std::vector<std::shared_ptr<MaterialGroup>>& mgs = r->getMaterialGroups();

			CUDABuffers cbs;
			try {
				//create cuda vertex buffer
				for (int i = 0; i < m_VertexAttributes.size(); ++i) {

					CUDABuffer cb;
					unsigned int glBufferID = v->getBufferID(m_VertexAttributes[i]);
					if (glBufferID != 0 ) {
						CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cb.cgr, glBufferID, cudaGraphicsRegisterFlagsReadOnly));
						CUDA_CHECK(cudaGraphicsMapResources(1, &cb.cgr, RTRenderer::getOptixStream()));
						CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&cb.memPtr, &size, cb.cgr));
//						cbs.vertexBuffers[i] = cb;
//						cbs.vertexBuffers[i].sizeInBytes = size;
//						cbs.vertexBuffers[i].numElements = (int)size / (4 * sizeof(float));
						cbs.vertexBuffers[m_VertexAttributes[i]] = cb;
						cbs.vertexBuffers[m_VertexAttributes[i]].sizeInBytes = size;
						cbs.vertexBuffers[m_VertexAttributes[i]].numElements = (int)size / (4 * sizeof(float));
					}
				}

				for (auto mg : mgs) {

					if (mg->getNumberOfPrimitives() > 0 && m_CudaVBOs.count(mg->getName()) == 0) {

						meshCount++;

						CUDABuffer ci;

						unsigned int glIndexID = mg->getIndexData()->getBufferID();
						// size on bytes
						size_t indexSize = mg->getIndexData()->getIndexSize() * sizeof(unsigned int);
						CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&ci.cgr, glIndexID, cudaGraphicsRegisterFlagsReadOnly));
						CUDA_CHECK(cudaGraphicsMapResources(1, &ci.cgr, RTRenderer::getOptixStream()));
						CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&ci.memPtr, &size, ci.cgr));
						ci.numElements = (int)(indexSize / (3 *sizeof(unsigned int)));
						ci.sizeInBytes = indexSize;
						cbs.indexBuffers[mg->getMaterialName()] = ci;
					}
				}
			}
			catch (std::exception const& e) {
				SLOG("Exception in creating CUDA buffers: %s", e.what());
					SLOG("Exception creating buffer for  scene object %s", sObj->getName().c_str());
				return false;
			}
			m_CudaVBOs[sObj->getName()] = cbs;	
		}
	}



	try {
		CUDA_SYNC_CHECK();
		//OptixTraversableHandle asHandle{ 0 };

		std::vector<OptixBuildInput> triangleInput(meshCount);
		std::vector<uint32_t> triangleInputFlags(meshCount);

		int meshID = 0;
		for (auto &scObj : m_CudaVBOs) {

			for (auto &indexBuffer : scObj.second.indexBuffers) {

				triangleInput[meshID] = {};
				triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

				triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(float4);
				triangleInput[meshID].triangleArray.numVertices = scObj.second.vertexBuffers[0].numElements;
				triangleInput[meshID].triangleArray.vertexBuffers = &(scObj.second.vertexBuffers[0].memPtr);//&memPtrVB[meshID];// &d_vertices[meshID];//&;

				triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(uint3);
				triangleInput[meshID].triangleArray.numIndexTriplets = indexBuffer.second.numElements ;//(int)model[meshID].index.size();
				triangleInput[meshID].triangleArray.indexBuffer = indexBuffer.second.memPtr;// memPtrI[meshID];//d_indices[meshID];

				triangleInputFlags[meshID] = 0;

				// in this example we have one SBT entry, and no per-primitive
				// materials:
				triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
				triangleInput[meshID].triangleArray.numSbtRecords = 1;
				triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
				triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
				triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;

				meshID++;
			}
		}


		// ==================================================================
		// BLAS setup
		// ==================================================================

		OptixAccelBuildOptions accelOptions = {0};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE| OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accelOptions.motionOptions.numKeys = 0;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes blasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage
		(RTRenderer::getOptixContext(),
			&accelOptions,
			triangleInput.data(),
			meshCount,  // num_build_inputs
			&blasBufferSizes
		));
		CUDA_SYNC_CHECK();

		// ==================================================================
		// prepare compaction
		// ==================================================================

		RTBuffer compactedSizeBuffer;
		compactedSizeBuffer.setSize(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.getPtr();

		// ==================================================================
		// execute build (main stage)
		// ==================================================================

		RTBuffer tempBuffer;
		tempBuffer.setSize(blasBufferSizes.tempSizeInBytes);

		RTBuffer outputBuffer;
		outputBuffer.setSize(blasBufferSizes.outputSizeInBytes);

		CUDA_SYNC_CHECK();
		
		OPTIX_CHECK(optixAccelBuild(RTRenderer::getOptixContext(),
			/* stream */0, 
			//RTRenderer::getOptixStream(),
			&accelOptions,
			triangleInput.data(),
			meshCount,
			tempBuffer.getPtr(),
			tempBuffer.getSize(),

			outputBuffer.getPtr(),
			outputBuffer.getSize(),

			&rtTraversableHandle,
			
			&emitDesc, 1
		));
		CUDA_SYNC_CHECK();

		// ==================================================================
		// perform compaction
		// ==================================================================
		uint64_t compactedSize;
		compactedSizeBuffer.getData(&compactedSize, sizeof(uint64_t));

		RTBuffer asBuffer;
		asBuffer.setSize(compactedSize);
		OPTIX_CHECK(optixAccelCompact(RTRenderer::getOptixContext(),
			/*stream:*/0,
			rtTraversableHandle,
			asBuffer.getPtr(),
			asBuffer.getSize(),
			&rtTraversableHandle));
		CUDA_SYNC_CHECK();

		// ==================================================================
		// aaaaaand .... clean up
		// ==================================================================
		outputBuffer.clear(); // << the UNcompacted, temporary output buffer
		tempBuffer.clear();
		compactedSizeBuffer.clear();

	}
	catch (std::exception const& e) {
		SLOG("Exception building acceleration structure: %s", e.what());
		return false;
	}

	return true;
}


#endif