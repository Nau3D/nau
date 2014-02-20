#ifndef OPTIXBUFFERLIB_H
#define OPTIXBUFFERLIB_H

#include <map>
#include <string>

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

namespace nau {
  namespace render {
   namespace optixRender {
		
		
	class OptixBufferLib {

	public:

		OptixBufferLib();
		void setContext(optix::Context &c);
		optix::Buffer &getBuffer(int glBufferID, unsigned int size);
		optix::Buffer &getIndexBuffer(int glBufferID, unsigned int size);

	private:

		optix::Context m_Context;
		std::map<unsigned int, optix::Buffer>  m_Buffer;
		std::map<unsigned int, optix::Buffer>  m_IndexBuffer;

	};
   };
  };
};


#endif