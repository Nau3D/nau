#ifndef MATERIALGROUP_H
#define MATERIALGROUP_H

#include "nau/geometry/indexData.h"
#include "nau/render/iRenderable.h"

#include <string>

namespace nau
{
	namespace material
	{

		class MaterialGroup 
		{
		public:

			static std::shared_ptr<MaterialGroup> Create(IRenderable *parent, std::string materialName);

			const std::string& getMaterialName ();
			void setMaterialName (std::string name);
		   
			unsigned int getNumberOfPrimitives(void);

			std::shared_ptr<nau::geometry::IndexData>& getIndexData (void);
			size_t getIndexOffset(void);
			size_t getIndexSize(void);

			void setParent (std::shared_ptr<nau::render::IRenderable> &parent);
			nau::render::IRenderable& getParent ();

			void updateIndexDataName();

			void setIndexList(std::shared_ptr<std::vector<unsigned int>> &indices);

			std::string &getName();

			virtual void compile() = 0;
			virtual bool isCompiled() = 0;
			virtual void resetCompilationFlag() = 0;
			virtual void bind() = 0;
			virtual void unbind() = 0;

			~MaterialGroup();

		protected:
			IRenderable *m_Parent;
			std::string m_MaterialName;
			std::shared_ptr<nau::geometry::IndexData> m_IndexData;
			std::string m_Name;

			MaterialGroup();
			MaterialGroup(IRenderable *parent, std::string materialName);
		};

	};
};
#endif // MaterialGroup_H
