#ifdef NAU_OPTIX 

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

		public:
			PassOptixPrime(const std::string &passName);
			virtual ~PassOptixPrime();

			static Pass *Create(const std::string &name);

			virtual void prepare (void);
			virtual void restore (void);

			//virtual bool renderTest (void);

			virtual void doPass (void);

			bool setQueryType(std::string);
			void addRayBuffer(IBuffer *b);
			void addHitBuffer(IBuffer *b);
		
		protected:
			static bool Init();
			static bool Inited;

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

	
	
