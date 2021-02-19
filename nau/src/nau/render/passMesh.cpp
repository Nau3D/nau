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
m_Buffer(0), m_Offset(0) {

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

	RENDERER->setMaterial(m_Mat);
	//m_Mat->prepare();	

	if (m_Buffer) {
		m_Buffer->getData(m_Offset, 4, &m_UIntProps[DIM_X]);
	}
	setupCamera();
	setupLights();

}


void
PassMesh::restore (void) {

	m_Mat->restore();
}


void
PassMesh::doPass (void) {

	PROFILE_GL("Task-Mesh shader");
	RENDERER->drawMeshTasks(m_Offset, m_UIntProps[DIM_X]);
}


void 
PassMesh::setMaterialName(const std::string &lName,const std::string &mName) {

	m_Mat = MATERIALLIBMANAGER->getMaterial(lName, mName);
	m_MaterialMap[mName] = MaterialID(lName, mName);
}


std::shared_ptr<Material> &
PassMesh::getMaterial() {

	return m_Mat;
}


void
PassMesh::setDimension(unsigned int dimX) {

	m_UIntProps[DIM_X] = dimX;
}


void
PassMesh::setDimFromBuffer(IBuffer  *buffNameX, unsigned int offX) {

	m_Buffer = buffNameX;
	m_Offset = offX;
}