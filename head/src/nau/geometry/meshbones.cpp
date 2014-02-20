#include <nau/geometry/meshbones.h>
#include <nau/material/imaterialgroup.h>
#include <nau/material/materialgroup.h>
#include <nau/render/irenderable.h>
#include <nau/render/vertexdata.h>
#include <nau/math/vec3.h>

//#include <algorithm>

using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;
using namespace nau::math;

MeshBones::MeshBones(void) : Mesh()
{
}

MeshBones::~MeshBones(void)
{
}


void 
MeshBones::addBoneWeight(unsigned int vertex, unsigned int bone, float weight)
{
	if (!m_BoneAssignments.count(vertex))
		m_BoneAssignments[vertex] = BoneWeights();

	m_BoneAssignments[vertex].push_back(std::pair<unsigned int, float>(bone,weight));
	
}

std::string 
MeshBones::getType()
{
	return("MeshBones");
}