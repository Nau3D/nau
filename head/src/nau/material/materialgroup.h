#ifndef MATERIALGROUP_H
#define MATERIALGROUP_H

#include <nau/render/irenderable.h>
#include <nau/render/indexdata.h>
#include <nau/material/imaterialgroup.h>

#include <string>

namespace nau
{
	namespace material
	{

		class MaterialGroup : public IMaterialGroup
		{
		public:
			MaterialGroup();
			virtual ~MaterialGroup();

			virtual const std::string& getMaterialName ();
			virtual void setMaterialName (std::string name);
		   
			virtual nau::render::IndexData& getIndexData (void);
			virtual void setIndexList (std::vector<unsigned int>* indices);
			virtual unsigned int getNumberOfPrimitives(void);


			virtual void setParent (nau::render::IRenderable* parent);
			virtual nau::render::IRenderable& getParent ();

		protected:
			nau::render::IRenderable* m_Parent;
			std::string m_MaterialName;
			nau::render::IndexData *m_IndexData;


		};

	};
};
#endif // MaterialGroup_H
