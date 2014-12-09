/*

This imaterialgroup subclass contains an index vector

*/

#ifndef MATERIALGROUP_H
#define MATERIALGROUP_H

#include <string>

#include <nau/render/irenderable.h>
#include <nau/render/indexdata.h>
#include <nau/material/imaterialgroup.h>



namespace nau
{
	namespace material
	{

		class MaterialGroup : public IMaterialGroup
		{
		public:
			MaterialGroup(IRenderable *parent, std::string materialName);
			~MaterialGroup();

			const std::string& getMaterialName ();
			void setMaterialName (std::string name);
		   
			unsigned int getNumberOfPrimitives(void);

			nau::render::IndexData& getIndexData (void);
			unsigned int getIndexOffset(void);
			unsigned int getIndexSize(void);

			void setParent (nau::render::IRenderable* parent);
			nau::render::IRenderable& getParent ();

			void setIndexList(std::vector<unsigned int>* indices);

			std::string getName();

		protected:
			nau::render::IRenderable* m_Parent;
			std::string m_MaterialName;
			nau::render::IndexData *m_IndexData;
			std::string m_Name;

			MaterialGroup();
		};

	};
};
#endif // MaterialGroup_H
