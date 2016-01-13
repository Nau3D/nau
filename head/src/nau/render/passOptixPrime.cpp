#ifdef NAU_OPTIX 


#include "nau.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
#include "nau/geometry/frustum.h"
#include "nau/render/passFactory.h"
#include "nau/render/passOptixPrime.h"

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <sstream>

using namespace nau::material;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::geometry;

//#ifdef DEBUG
#define CHK_PRIME( code )                                                      \
{                                                                              \
  RTPresult res__ = code;                                                      \
  if( res__ != RTP_SUCCESS )                                                   \
    {                                                                            \
    const char* err_string;                                                    \
	  rtpContextGetLastErrorString( m_Context, &err_string );\
    std::cerr << "Error on line " << __LINE__ << ": '"                         \
              << err_string                                                    \
              << "' (" << res__ << ")" << std::endl;                           \
			  cudaDeviceReset();\
    exit(1);                                                                   \
    }                                                                            \
}
//#else
//#define CHK_PRIME(code) code
//#endif

static const char *_cudaGetErrorEnum(cudaError_t error)
{
	switch (error)
	{
	case cudaSuccess:
		return "cudaSuccess";

	case cudaErrorMissingConfiguration:
		return "cudaErrorMissingConfiguration";

	case cudaErrorMemoryAllocation:
		return "cudaErrorMemoryAllocation";

	case cudaErrorInitializationError:
		return "cudaErrorInitializationError";

	case cudaErrorLaunchFailure:
		return "cudaErrorLaunchFailure";

	case cudaErrorPriorLaunchFailure:
		return "cudaErrorPriorLaunchFailure";

	case cudaErrorLaunchTimeout:
		return "cudaErrorLaunchTimeout";

	case cudaErrorLaunchOutOfResources:
		return "cudaErrorLaunchOutOfResources";

	case cudaErrorInvalidDeviceFunction:
		return "cudaErrorInvalidDeviceFunction";

	case cudaErrorInvalidConfiguration:
		return "cudaErrorInvalidConfiguration";

	case cudaErrorInvalidDevice:
		return "cudaErrorInvalidDevice";

	case cudaErrorInvalidValue:
		return "cudaErrorInvalidValue";

	case cudaErrorInvalidPitchValue:
		return "cudaErrorInvalidPitchValue";

	case cudaErrorInvalidSymbol:
		return "cudaErrorInvalidSymbol";

	case cudaErrorMapBufferObjectFailed:
		return "cudaErrorMapBufferObjectFailed";

	case cudaErrorUnmapBufferObjectFailed:
		return "cudaErrorUnmapBufferObjectFailed";

	case cudaErrorInvalidHostPointer:
		return "cudaErrorInvalidHostPointer";

	case cudaErrorInvalidDevicePointer:
		return "cudaErrorInvalidDevicePointer";

	case cudaErrorInvalidTexture:
		return "cudaErrorInvalidTexture";

	case cudaErrorInvalidTextureBinding:
		return "cudaErrorInvalidTextureBinding";

	case cudaErrorInvalidChannelDescriptor:
		return "cudaErrorInvalidChannelDescriptor";

	case cudaErrorInvalidMemcpyDirection:
		return "cudaErrorInvalidMemcpyDirection";

	case cudaErrorAddressOfConstant:
		return "cudaErrorAddressOfConstant";

	case cudaErrorTextureFetchFailed:
		return "cudaErrorTextureFetchFailed";

	case cudaErrorTextureNotBound:
		return "cudaErrorTextureNotBound";

	case cudaErrorSynchronizationError:
		return "cudaErrorSynchronizationError";

	case cudaErrorInvalidFilterSetting:
		return "cudaErrorInvalidFilterSetting";

	case cudaErrorInvalidNormSetting:
		return "cudaErrorInvalidNormSetting";

	case cudaErrorMixedDeviceExecution:
		return "cudaErrorMixedDeviceExecution";

	case cudaErrorCudartUnloading:
		return "cudaErrorCudartUnloading";

	case cudaErrorUnknown:
		return "cudaErrorUnknown";

	case cudaErrorNotYetImplemented:
		return "cudaErrorNotYetImplemented";

	case cudaErrorMemoryValueTooLarge:
		return "cudaErrorMemoryValueTooLarge";

	case cudaErrorInvalidResourceHandle:
		return "cudaErrorInvalidResourceHandle";

	case cudaErrorNotReady:
		return "cudaErrorNotReady";

	case cudaErrorInsufficientDriver:
		return "cudaErrorInsufficientDriver";

	case cudaErrorSetOnActiveProcess:
		return "cudaErrorSetOnActiveProcess";

	case cudaErrorInvalidSurface:
		return "cudaErrorInvalidSurface";

	case cudaErrorNoDevice:
		return "cudaErrorNoDevice";

	case cudaErrorECCUncorrectable:
		return "cudaErrorECCUncorrectable";

	case cudaErrorSharedObjectSymbolNotFound:
		return "cudaErrorSharedObjectSymbolNotFound";

	case cudaErrorSharedObjectInitFailed:
		return "cudaErrorSharedObjectInitFailed";

	case cudaErrorUnsupportedLimit:
		return "cudaErrorUnsupportedLimit";

	case cudaErrorDuplicateVariableName:
		return "cudaErrorDuplicateVariableName";

	case cudaErrorDuplicateTextureName:
		return "cudaErrorDuplicateTextureName";

	case cudaErrorDuplicateSurfaceName:
		return "cudaErrorDuplicateSurfaceName";

	case cudaErrorDevicesUnavailable:
		return "cudaErrorDevicesUnavailable";

	case cudaErrorInvalidKernelImage:
		return "cudaErrorInvalidKernelImage";

	case cudaErrorNoKernelImageForDevice:
		return "cudaErrorNoKernelImageForDevice";

	case cudaErrorIncompatibleDriverContext:
		return "cudaErrorIncompatibleDriverContext";

	case cudaErrorPeerAccessAlreadyEnabled:
		return "cudaErrorPeerAccessAlreadyEnabled";

	case cudaErrorPeerAccessNotEnabled:
		return "cudaErrorPeerAccessNotEnabled";

	case cudaErrorDeviceAlreadyInUse:
		return "cudaErrorDeviceAlreadyInUse";

	case cudaErrorProfilerDisabled:
		return "cudaErrorProfilerDisabled";

	case cudaErrorProfilerNotInitialized:
		return "cudaErrorProfilerNotInitialized";

	case cudaErrorProfilerAlreadyStarted:
		return "cudaErrorProfilerAlreadyStarted";

	case cudaErrorProfilerAlreadyStopped:
		return "cudaErrorProfilerAlreadyStopped";

#if __CUDA_API_VERSION >= 0x4000

	case cudaErrorAssert:
		return "cudaErrorAssert";

	case cudaErrorTooManyPeers:
		return "cudaErrorTooManyPeers";

	case cudaErrorHostMemoryAlreadyRegistered:
		return "cudaErrorHostMemoryAlreadyRegistered";

	case cudaErrorHostMemoryNotRegistered:
		return "cudaErrorHostMemoryNotRegistered";
#endif

	case cudaErrorStartupFailure:
		return "cudaErrorStartupFailure";

	case cudaErrorApiFailureBase:
		return "cudaErrorApiFailureBase";
	}

	return "<unknown>";
}
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		SLOG("CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		//cudaDeviceReset();
			// Make sure we call CUDA Device Reset before exiting
			//exit(EXIT_FAILURE);
	}
}


// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )


bool PassOptixPrime::Inited = PassOptixPrime::Init();


bool
PassOptixPrime::Init() {

	Attribs.add(Attribute(RAY_COUNT, "RAY_COUNT", Enums::INT, false, new NauInt(-1)));
#ifndef _WINDLL
	NAU->registerAttributes("PASS", &Attribs);
#endif

	PASSFACTORY->registerClass("optixPrime", Create);

	return true;
}


PassOptixPrime::PassOptixPrime(const std::string &passName) : Pass(passName) {

	m_ClassName = "optix prime";
	m_Context = NULL;
	m_RayCountBuffer = NULL;
}


PassOptixPrime::~PassOptixPrime() {

	if (m_Context)
		CHK_PRIME(rtpContextDestroy(m_Context));
}


std::shared_ptr<Pass>
PassOptixPrime::Create(const std::string &passName) {

	return dynamic_pointer_cast<Pass>(std::shared_ptr<PassOptixPrime>(new PassOptixPrime(passName)));
}


void
PassOptixPrime::prepare(void) {


	int k;
	if (!m_Init)
		initOptixPrime();
	else {
//		checkCudaErrors(cudaGraphicsMapResources(1, &cgl, 0));
////		k = cudaGraphicsMapResources(1, &cglInd, 0);
		k = cudaGraphicsMapResources(1, &cglBuff, 0);
		k = cudaGraphicsMapResources(1, &cglBuffH, 0);
		k = cudaGraphicsMapResources(1, &cglInd, 0);
	}

}


void
PassOptixPrime::restore(void) {

	int k;
//	k = cudaGraphicsUnmapResources(1, &cgl, 0);
//	cudaGraphicsUnmapResources(1, &cglInd, 0);
	k = cudaGraphicsUnmapResources(1, &cglBuff, 0);
	k = cudaGraphicsUnmapResources(1, &cglBuffH, 0);
	k = cudaGraphicsUnmapResources(1, &cglInd, 0);
	//SLOG("%d", k);
}



void
PassOptixPrime::doPass(void) {

	if (m_RayCountBuffer) {
		m_RayCountBuffer->getData(m_RayCountBufferOffset, sizeof(int), &m_IntProps[RAY_COUNT]);
	}

	if (m_IntProps[RAY_COUNT] == -1) {
		m_IntProps[RAY_COUNT] = m_Hits->getPropui(IBuffer::SIZE) / 16;
	}
	CHK_PRIME(rtpBufferDescSetRange(m_RaysDesc, 0, m_IntProps[RAY_COUNT]));
	CHK_PRIME(rtpBufferDescSetRange(m_HitsDesc, 0, m_IntProps[RAY_COUNT]));
	CHK_PRIME(rtpQuerySetRays(m_Query, m_RaysDesc));
	CHK_PRIME(rtpQuerySetHits(m_Query, m_HitsDesc));
	const char *err_string;
	rtpContextGetLastErrorString(m_Context, &err_string);
	CHK_PRIME(rtpQueryExecute(m_Query, 0 /* hints */));
	//CHK_PRIME(rtpQueryFinish(m_Query));
}


void 
PassOptixPrime::initOptixPrime() {
	
	// Create Context
	CHK_PRIME(rtpContextCreate(RTP_CONTEXT_TYPE_CUDA, &m_Context));

	// Create Vertex Buffer
	std::shared_ptr<IRenderable> &renderable = RENDERMANAGER->getScene(m_SceneVector[0])->getSceneObject(0)->getRenderable();
	int vbo = renderable->getVertexData()->getBufferID(0);
	int numVert = renderable->getVertexData()->getNumberOfVertices();
	//std::shared_ptr<std::vector<VertexAttrib>> &vertex = renderable->getVertexData()->getDataOf(0);

	size_t size;
	void * devPtr;
	int k = cudaGraphicsGLRegisterBuffer(&cgl, vbo, cudaGraphicsRegisterFlagsReadOnly);
	k = cudaGraphicsMapResources(1, &cgl, 0);
	k = cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size, cgl);

	CHK_PRIME(rtpBufferDescCreate(
		m_Context,
		RTP_BUFFER_FORMAT_VERTEX_FLOAT4,
		RTP_BUFFER_TYPE_CUDA_LINEAR,
		devPtr,
		&m_VerticesDesc)
		);
	CHK_PRIME(rtpBufferDescSetRange(m_VerticesDesc, 0, numVert));

	// Create Index Buffer
	std::shared_ptr<IndexData> &ind = renderable->getIndexData();
	std::vector<int> v; 
	ind->getIndexDataAsInt(&v);
	GLuint index;
	IBuffer *b;
	b = RESOURCEMANAGER->createBuffer(m_Name);
	index = b->getPropi(IBuffer::ID);
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, v.size() * sizeof(int), &(v)[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	int numInd = (int)v.size();

	void * devPtrInd;
	k = cudaGraphicsGLRegisterBuffer(&cglInd, index, cudaGraphicsRegisterFlagsReadOnly);
	k = cudaGraphicsMapResources(1, &cglInd, 0);
	k = cudaGraphicsResourceGetMappedPointer((void **)&devPtrInd, &size, cglInd);
	CHK_PRIME(rtpBufferDescCreate(
		m_Context,
		RTP_BUFFER_FORMAT_INDICES_INT3,
		RTP_BUFFER_TYPE_CUDA_LINEAR, 
		devPtrInd,
		&m_IndicesDesc)
		);
	CHK_PRIME(rtpBufferDescSetRange(m_IndicesDesc, 0, numInd/3));

	// Create Model
	CHK_PRIME(rtpModelCreate(m_Context, &m_Model));
	CHK_PRIME(rtpModelSetTriangles(m_Model, m_IndicesDesc, m_VerticesDesc));
	int useCallerTris = 0;
	CHK_PRIME(rtpModelSetBuilderParameter(m_Model, RTP_BUILDER_PARAM_USE_CALLER_TRIANGLES, sizeof(int), &useCallerTris));
	CHK_PRIME(rtpModelUpdate(m_Model, 0));
	CHK_PRIME(rtpModelFinish(m_Model));

	cudaGraphicsUnmapResources(1, &cgl, 0);

	// Create Rays Buffer
	int rayBufferID = m_Rays->getPropi(IBuffer::ID);
	int rayBufferRayCount = m_Rays->getPropui(IBuffer::SIZE) / (8 * sizeof(float));

	void * devPtrBuff;
	k = cudaGraphicsGLRegisterBuffer(&cglBuff, rayBufferID, cudaGraphicsRegisterFlagsReadOnly);
	k = cudaGraphicsMapResources(1, &cglBuff, 0);
	k = cudaGraphicsResourceGetMappedPointer((void **)&devPtrBuff, &size, cglBuff);

	CHK_PRIME(rtpBufferDescCreate(
		m_Context,
		RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX,
		RTP_BUFFER_TYPE_CUDA_LINEAR,
		devPtrBuff,
		&m_RaysDesc)
		);
	CHK_PRIME(rtpBufferDescSetRange(m_RaysDesc, 0, rayBufferRayCount));

	
	// Create Hits Buffer
	int hitBufferID = m_Hits->getPropi(IBuffer::ID);
	void * devPtrBuffH;
	k = cudaGraphicsGLRegisterBuffer(&cglBuffH, hitBufferID, cudaGraphicsRegisterFlagsWriteDiscard);
	k = cudaGraphicsMapResources(1, &cglBuffH, 0);
	k = cudaGraphicsResourceGetMappedPointer((void **)&devPtrBuffH, &size, cglBuffH);

	CHK_PRIME(rtpBufferDescCreate(
		m_Context,
		RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V,
		RTP_BUFFER_TYPE_CUDA_LINEAR,
		devPtrBuffH,
		&m_HitsDesc)
		);
	CHK_PRIME(rtpBufferDescSetRange(m_HitsDesc, 0, rayBufferRayCount));

	// Prepare query
	CHK_PRIME(rtpQueryCreate(m_Model, m_QueryType, &m_Query));
	CHK_PRIME(rtpQuerySetRays(m_Query, m_RaysDesc));
	CHK_PRIME(rtpQuerySetHits(m_Query, m_HitsDesc));

	m_Init = true;
}


bool
PassOptixPrime::setQueryType(std::string qt) {

	bool b = true;
	if ("CLOSEST" == qt)
		m_QueryType = RTP_QUERY_TYPE_CLOSEST;
	else if ("ANY" == qt)
		m_QueryType = RTP_QUERY_TYPE_ANY;
	else
		b = false;

	return b;
}


void 
PassOptixPrime::addHitBuffer(IBuffer *b) {

	m_Hits = b;
}

void 
PassOptixPrime::setBufferForRayCount(IBuffer * b, unsigned int offset) {

	m_RayCountBuffer = b;
	m_RayCountBufferOffset = offset;
}


void
PassOptixPrime::addRayBuffer(IBuffer *b) {

	m_Rays = b;
}

#endif
