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
			//friend class boost::serialization::access;
		public:
			MaterialGroup();
			virtual ~MaterialGroup();

		//	virtual int getMaterialId ();
		//	virtual void setMaterialId (int id);

			virtual const std::string& getMaterialName ();
			virtual void setMaterialName (std::string name);
		   
			//virtual std::vector<unsigned int>& getIndexList ();

			virtual nau::render::IndexData& getIndexData (void);
			virtual void setIndexList (std::vector<unsigned int>* indices);
			virtual unsigned int getNumberOfPrimitives(void);


			virtual void setParent (nau::render::IRenderable* parent);
			virtual nau::render::IRenderable& getParent ();

			// FIXME: This has to go away
			// see also: cworldfactory.cpp
			// ---CUT HERE---
			virtual void bakeMayaUVTextureProfile (float repeat_u, float repeat_v,
								 float coverage_u, float coverage_v);
			// ---CUT HERE---

		protected:
			//int m_MaterialId;
			nau::render::IRenderable* m_Parent;
			std::string m_MaterialName;
			//std::vector<unsigned int> *m_IndexList;
			nau::render::IndexData *m_IndexData; /* change name to reflect that this isreally an index array */

		public:
			/*
			* boost serialization interface
			*/
			//template<class Archive>
			//void serialize (Archive &ar, const unsigned int version)
			//{
			//	ar & m_MaterialId;
			//	ar & m_MaterialName;
			//	ar & m_IndexList;
			//}

		};

	};
};
#endif // MaterialGroup_H
