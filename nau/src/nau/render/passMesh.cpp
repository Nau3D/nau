#include "nau/render/passMesh.h"

#include "nau.h"
#include "nau/debug/profile.h"
#include "nau/render/passFactory.h"

using namespace nau::material;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::geometry;

bool PassMesh::Inited = PassMesh::Init();

bool
PassMesh::Init() {

	PASSFACTORY->registerClass("mesh", Create);

	return true;
}


PassMesh::PassMesh(const std::string &passName) : Pass(passName),
m_Buffer(0), m_ActiveMat(0) {

	m_ClassName = "mesh";

	registerAndInitArrays(Attribs);
}


PassMesh::~PassMesh() {

}


std::shared_ptr<Pass>
PassMesh::Create(const std::string &passName) {

	return dynamic_pointer_cast<Pass>(std::shared_ptr<PassMesh>(new PassMesh(passName)));
}


void
PassMesh::eventReceived(const std::string &sender, const std::string &eventType,
	const std::shared_ptr<IEventData> &evt) {

}


void
PassMesh::prepare (void) {

	//m_Mat->prepare();	

	
	setupCamera();
	setupLights();

}


void
PassMesh::restore (void) {

	m_Mat[0].material->restore();
}


void
PassMesh::doPass (void) {

	PROFILE_GL("Task-Mesh shader");

	unsigned int count;
	for (auto &m : m_Mat) {
		m_ActiveMat = m.material;
		if (m_Buffer) {
			m_Buffer->getData(m.offset, 4, &count);
		}
		else count = m.count;
		RENDERER->setMaterial(m.material);

		RENDERER->drawMeshTasks(0, m.count);
	}
}


void 
PassMesh::addMaterial(const std::string &lName,const std::string &mName, unsigned int count, IBuffer* buf, unsigned int offset) {

	materials m;

	m.material = MATERIALLIBMANAGER->getMaterial(lName, mName);
	m.count = count;
	m.offset = offset;

	m_Buffer = buf;

	m_Mat.push_back(m);
	m_MaterialMap[mName] = MaterialID(lName, mName);
}


std::shared_ptr<Material> &
PassMesh::getMaterial() {

	return m_ActiveMat;
}


//void
//PassMesh::setDimension(unsigned int dimX) {
//
//	m_UIntProps[DIM_X] = dimX;
//}
//
//
//void
//PassMesh::setDimFromBuffer(IBuffer  *buffNameX, unsigned int offX) {
//
//	m_Buffer = buffNameX;
//	m_Offset = offX;
//}