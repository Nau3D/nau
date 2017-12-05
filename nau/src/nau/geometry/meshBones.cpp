#include "nau/geometry/meshBones.h"

#include "nau/material/materialGroup.h"
#include "nau/render/iRenderable.h"
#include "nau/geometry/vertexData.h"
#include "nau/math/vec3.h"


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
MeshBones::getClassName() {

	return "MeshBones";
}


