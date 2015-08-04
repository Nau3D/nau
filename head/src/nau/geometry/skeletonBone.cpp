#include "nau/geometry/skeletonBone.h"

using namespace nau::geometry;

SkeletonBone::SkeletonBone():
		m_Angle(0),
		m_Position(0.0f, 0.0f, 0.0f),
		m_RotVector(0.0f, 1.0f, 0.0f),
		m_Id(0),
		m_Name(""),
		m_LocalTransform(),
		m_CompositeTransform()
{}

		
SkeletonBone::SkeletonBone(std::string name, unsigned int id, vec3 pos, float angle, vec3 axis):
		m_Name(name),
		m_Id(id),
		m_Position(pos.x, pos.y, pos.z),
		m_Angle(angle),
		m_RotVector(axis.x, axis.y, axis.z),
		m_LocalTransform(),
		m_CompositeTransform()
{}

SkeletonBone::~SkeletonBone() 
{}

void 
SkeletonBone::setPosition(vec3 pos) 
{
	m_Position.set(pos.x, pos.y, pos.z);
	// Still have to set the transform
}

void 
SkeletonBone::setRotation(vec3 axis, float angle) 
{
	m_Angle = angle;
	m_RotVector.set(axis.x,axis.y, axis.z);
	// still have to set the transform
}

void
SkeletonBone::setId(unsigned int i)
{
	m_Id = i;
}

void 
SkeletonBone::setName(std::string name) 
{
	m_Name = name;
}

mat4 &
SkeletonBone::getFullTransform() 
{
	return m_CompositeTransform;
}

mat4 &
SkeletonBone::getLocalTransform() 
{
	return m_LocalTransform;
}

unsigned int
SkeletonBone::getID()
{
	return m_Id;
}

std::string
SkeletonBone::getName()
{
	return m_Name;
}