#if NAU_OPTIX == 1

#ifndef PASSOPTIXPRIME_H
#define PASSOPTIXPRIME_H

#include "nau/material/iBuffer.h"
#include "nau/render/pass.h"

//#include <GL/glew.h>
#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
#include <optix_prime/optix_prime.h>

#include <sstream>



namespace nau
{
	namespace render
	{

		class PassOptixPrime : public Pass {
			friend class PassFactory;

		public:
			virtual ~PassOptixPrime();

			INT_PROP(RAY_COUNT, 201);


			static std::shared_ptr<Pass> Create(const std::string &name);

			virtual void prepare (void);
			virtual void restore (void);

			//virtual bool renderTest (void);

			virtual void doPass (void);

			bool setQueryType(std::string);
			void addRayBuffer(IBuffer *b);
			void addHitBuffer(IBuffer *b);
			void setBufferForRayCount(IBuffer *b, unsigned int offset);
		
		protected:
			PassOptixPrime(const std::string &passName);
			static bool Init();
			static bool Inited;

			IBuffer *m_RayCountBuffer;
			unsigned int m_RayCountBufferOffset;
			bool m_Init = false;
			IBuffer *m_Rays = NULL, *m_Hits = NULL;
			RTPcontext m_Context;
			RTPquery m_Query;
			RTPbufferdesc m_VerticesDesc, m_IndicesDesc, m_RaysDesc, m_HitsDesc;
			RTPquerytype m_QueryType = RTP_QUERY_TYPE_ANY;
			RTPmodel m_Model;
			cudaGraphicsResource *cgl, *cglInd, *cglBuff, *cglBuffH;

			void initOptixPrime();
		};
	};
};
#endif


#endif

	
	
