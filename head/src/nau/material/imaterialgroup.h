#ifndef IMATERIALGROUP_H
#define IMATERIALGROUP_H

#include <vector>
#include <nau/render/indexdata.h>



//Forward declaration. It would be VERY nice if this could be removed
namespace nau
{
	namespace render
	{
		class IRenderable;
	};
};

namespace nau
{
	namespace material
	{
		class IMaterialGroup
		{
		public:
			//virtual int getMaterialId () = 0;
			//virtual void setMaterialId (int id) = 0;
			virtual const std::string & getMaterialName () = 0;
			virtual void setMaterialName (std::string name) = 0;
			virtual unsigned int getNumberOfPrimitives(void) = 0;

			
			virtual nau::render::IndexData& getIndexData (void) = 0;
			//virtual std::vector<unsigned int>& getIndexList () = 0;

			virtual void setParent (nau::render::IRenderable* parent) = 0;
			virtual nau::render::IRenderable& getParent () = 0;
		   
			// FIXME: This has to go away
			// see also: cworldfactory.cpp
			virtual void bakeMayaUVTextureProfile (float repeat_u, float repeat_v,
								 float coverage_u, float coverage_v) = 0;
		   
			virtual ~IMaterialGroup () {}
		};
	};
};

#endif
