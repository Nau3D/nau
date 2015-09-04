#include "nau/scene/sceneSkeleton.h"

using namespace nau::scene;

SceneSkeleton::SceneSkeleton()
{}

SceneSkeleton::~SceneSkeleton()
{}


std::string
SceneSkeleton::getType()
{
	return "SceneSkeleton";
}


void
SceneSkeleton::compile()
{
}



void 
SceneSkeleton::eventReceived(const std::string &sender, const std::string &eventType, IEventData *evt)
{}

void 
SceneSkeleton::addAnim(std::string aName, float aLength)
{
	m_Anims[aName] = SkeletonAnim();
	m_Anims[aName].setLength(aLength);
}


// must exist!
SkeletonAnim &
SceneSkeleton::getAnim(std::string name) 
{
	return(m_Anims[name]);
}

void 
SceneSkeleton::addBone(std::string name, unsigned int id, vec3 pos, float angle, vec3 axis) 
{
	m_Bones[id] = SkeletonBone(name, id, pos, angle, axis);
}

void 
SceneSkeleton::setBoneRelation(std::string child, std::string parent)
{
	int childId, parentId;

	childId = seekBoneID(child);
	parentId = seekBoneID(parent);

	if (childId == -1 || parentId == -1)
		return;

	m_BoneHierarchy[childId] = parentId;
}

int
SceneSkeleton::seekBoneID(std::string name)
{
	std::map<unsigned int, SkeletonBone>::iterator iter;

	iter = m_Bones.begin();

	for( ; iter != m_Bones.end(); ++iter) {
	
		if ((*iter).second.getName() == name)
			return ((*iter).second.getID());
	}
	return(-1);
}
