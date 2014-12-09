
#include <nau/geometry/meshwithpose.h>
#include <nau.h>
#include <nau/slogger.h>

MeshPose::MeshPose(void): Mesh(),
		m_ActivePose(0)
{
	EVENTMANAGER->addListener("NEXT_POSE",(nau::event_::IListener *)this);
}

MeshPose::~MeshPose(void) 
{
	EVENTMANAGER->removeListener("NEXT_POSE",this);

	std::vector<PoseOffset *>::iterator	poseIter = m_vOffsets.begin();

	for ( ; poseIter != m_vOffsets.end(); poseIter++) {

		delete (*poseIter);
	}
}


void 
MeshPose::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
	if(eventType=="NEXT_POSE") {
		m_ActivePose++;
		if (m_ActivePose >= m_vOffsets.size())
			m_ActivePose = 0;
		setPose(m_ActivePose);
	}
}

void
MeshPose::setReferencePose(std::vector<vec4> vertexData)
{
	m_ReferencePose.resize(vertexData.size());

	for ( unsigned int i = 0 ; i < vertexData.size(); i++) {
		m_ReferencePose[i].set(vertexData[i].x, vertexData[i].y, vertexData[i].z);
	}
}

void 
MeshPose::addPose(std::string aName, PoseOffset *aPose) 
{
	m_vOffsets.push_back(aPose);
	m_vNames.push_back(aName);
}


unsigned int 
MeshPose::getActivePose() {

	return m_ActivePose;
}

void 
MeshPose::setPose(unsigned int index) 
{
	// this allows for easy cycling between poses
	if (index >= m_vOffsets.size())
		index = index % m_vOffsets.size();

	m_ActivePose = index;

	// if there are offsets
	if (m_vOffsets.size()) {
	
		// have to add Pose offsets to the Reference pose
		// and set the result in the vertexdata
		std::vector<VertexData::Attr>& vertexData = m_pVertexData->getDataOf(VertexData::getAttribIndex("position"));
		std::vector<vec3> offsets = m_vOffsets[index]->getOffsets();

		vec3 v;

		for (unsigned int i = 0 ; i < vertexData.size(); i++) 
		{
			v.x = m_ReferencePose[i].x + offsets[i].x;
			v.y = m_ReferencePose[i].y + offsets[i].y;
			v.z = m_ReferencePose[i].z + offsets[i].z;

			vertexData[i].set(v.x, v.y, v.z);
		}
	}
	// else set the reference pose
	else {
		
		// similar to above, but without adding the offsets
		std::vector<VertexData::Attr>& vertexData = m_pVertexData->getDataOf(VertexData::getAttribIndex("position"));
		vec3 v;

		for (unsigned int i = 0 ; i < vertexData.size(); i++) 
		{
			v.x = m_ReferencePose[i].x;
			v.y = m_ReferencePose[i].y;
			v.z = m_ReferencePose[i].z;

			vertexData[i].set(v.x, v.y, v.z);
		}
	}
	resetCompilationFlags();
}

unsigned int
MeshPose::getNumberOfPoses() 
{
	return m_vOffsets.size();
}


void 
MeshPose::setPose(std::string aName)
{
	unsigned int index = 0;
	while (index < m_vNames.size() && m_vNames[index] != aName) {
		index++;
	}
	if (index < m_vNames.size()) {
		m_ActivePose = index;
		setPose(index);
	}

	resetCompilationFlags();

}


void 
MeshPose::setReferencePose() 
{
	std::vector<VertexData::Attr>& vertexData = m_pVertexData->getDataOf(VertexData::getAttribIndex("position"));

	vec3 v;

	std::map<unsigned int , float >::iterator iter;

	// set the vertex data to the reference pose
	for (unsigned int i = 0 ; i < vertexData.size(); i++) {
	
		v.x = m_ReferencePose[i].x;
		v.y = m_ReferencePose[i].y;
		v.z = m_ReferencePose[i].z;

		vertexData[i].set(v.x, v.y, v.z);
	}
	
	resetCompilationFlags();

}


void
MeshPose::setPose(std::map<unsigned int , float > *influences)
{
	// THIS IS WHERE THE MESH IS SET BASED ON THE POSE INFLUENCES

	std::vector<VertexData::Attr>& vertexData = m_pVertexData->getDataOf(VertexData::getAttribIndex("position"));

	vec3 v;

	std::map<unsigned int , float >::iterator iter;

	// first set the vertex data to the reference pose
	for (unsigned int i = 0 ; i < vertexData.size(); i++) {
	
		v.x = m_ReferencePose[i].x;
		v.y = m_ReferencePose[i].y;
		v.z = m_ReferencePose[i].z;

		vertexData[i].set(v.x, v.y, v.z);
	}

	// for each pose
	vec4 v2; 
	for (iter = influences->begin(); iter != influences->end(); ++iter) {
	//SLOG("Pose %d Influence %f", (*iter).first,(*iter).second);
		std::vector<vec3> offsets = m_vOffsets[(*iter).first]->getOffsets();

		for (unsigned int i = 0 ; i < vertexData.size(); i++) {

				v2.x = offsets[i].x * (*iter).second;
				v2.y = offsets[i].y * (*iter).second;
				v2.z = offsets[i].z * (*iter).second;
				v2.w = 0.0;
				vertexData[i].add(v2);
		}
	}
	resetCompilationFlags();
}

std::string 
MeshPose::getType() 
{
	return ("MeshPose");
}