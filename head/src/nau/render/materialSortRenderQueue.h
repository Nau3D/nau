#ifndef MATERIALSORTRENDERQUEUE_H
#define MATERIALSORTRENDERQUEUE_H

// long name warning
#ifdef WIN32
  #pragma warning(disable : 4503)
#endif


#include "nau/render/iRenderQueue.h"

#include "nau/material/material.h"
#include "nau/material/materialGroup.h"
#include "nau/material/materialId.h"
#include "nau/math/matrix.h"

namespace nau
{
	namespace render
	{
		class MaterialSortRenderQueue : public IRenderQueue 
		{
		friend class RenderQueueFactory;
		
		public:
			void clearQueue (void);
			void addToQueue (nau::scene::SceneObject* aObject,
				std::map<std::string, nau::material::MaterialID> &materialMap);
			void processQueue (void);
		protected:
			MaterialSortRenderQueue(void);
		public:
			~MaterialSortRenderQueue(void);

		private:
			std::map<int, 
					std::map<nau::material::Material*, 
							std::vector<std::pair<nau::material::MaterialGroup*, 
												  nau::math::mat4 *> >* >* > m_RenderQueue;
		};
	};
};

#endif //MATERIALSORTRENDERQUEUE_H
