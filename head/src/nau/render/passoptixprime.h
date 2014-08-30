#ifndef PASSOPTIXPRIME_H
#define PASSOPTIXPRIME_H

#include <sstream>
#include <GL/glew.h>
#include <optix_prime/optix_prime.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <nau/render/pass.h>
#include <nau/render/ibuffer.h>


namespace nau
{
	namespace render
	{

		class PassOptixPrime : public Pass {

		public:
			PassOptixPrime(const std::string &passName);
			virtual ~PassOptixPrime();

			virtual void prepare (void);
			virtual void restore (void);
			virtual bool renderTest (void);
			virtual void doPass (void);

			bool setQueryType(std::string);
			void addRayBuffer(IBuffer *b);
			void addHitBuffer(IBuffer *b);
		
		protected:
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





	
	
