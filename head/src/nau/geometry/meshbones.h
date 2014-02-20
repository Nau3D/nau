#ifndef MESHBONES_H
#define MESHBONES_H

#include <nau/geometry/mesh.h>
#include <nau/render/irenderer.h>
#include <nau/render/irenderable.h>
#include <nau/material/imaterialgroup.h>


#include <string>

namespace nau
{
	namespace geometry
	{
		class MeshBones : public Mesh
		{
		protected:
			// the weight for a bone
			typedef std::pair<unsigned int, float> BoneWeight;
			// the weights for all bones assigned to a vertex
			typedef std::vector<BoneWeight> BoneWeights;
			// the bone assignements for each vertex
			typedef std::map<unsigned int, BoneWeights> BoneAssignments;

			BoneAssignments m_BoneAssignments;

			MeshBones(void);

		public:

			friend class nau::resource::ResourceManager;
			~MeshBones (void);

			void addBoneWeight(unsigned int vertex, unsigned int bone, float weight);
			virtual std::string getType (void);
		};
	};
};

#endif
