#include "nau/render/optix/optixGeometry.h"
#include "nau/scene/sceneObject.h"
#include "nau.h"

using namespace nau::render::optixRender;

OptixGeometry::OptixGeometry() {

	m_VertexAttributes.resize(VertexData::MaxAttribs);
	for (int i = 0; i < VertexData::MaxAttribs; ++i) 
		m_VertexAttributes[i] = false;
}


void 
OptixGeometry::addVertexAttribute(unsigned int attr) {

	if (attr < VertexData::MaxAttribs)
		m_VertexAttributes[attr] = true;
}


void
OptixGeometry::setMaterialLib(OptixMaterialLib *oml) {

	m_MaterialLib = oml;
}


void
OptixGeometry::setContext(optix::Context &c) {

	m_Context = c;
	//m_GeomIntersect = m_Context->createProgramFromPTXFile("optix/common.ptx","geometryintersection");
	//m_BoundingBox = m_Context->createProgramFromPTXFile("optix/common.ptx","boundingbox");
}


void
OptixGeometry::setGeometryIntersectProc(std::string file, std::string proc) {

	m_GeomIntersect = m_Context->createProgramFromPTXFile(file,proc);
}


void
OptixGeometry::setBoundingBoxProc(std::string file, std::string proc) {

	m_BoundingBox = m_Context->createProgramFromPTXFile(file,proc);
}


void
OptixGeometry::setBufferLib(OptixBufferLib *obl) {

	m_BufferLib = obl;
}

#include "nau/slogger.h"

void
OptixGeometry::addSceneObject(SceneObject *s, std::map<std::string, nau::material::MaterialID> & materialMap) {
//OptixGeometry::addSceneObject(int id, std::map<std::string, nau::material::MaterialID> & materialMap) {

//	IRenderable &r = RENDERMANAGER->getSceneObject(id)->getRenderable();
	IRenderable &r = s->getRenderable();
	std::shared_ptr<VertexData> &v = r.getVertexData();
	size_t size = v->getDataOf(0)->size();

	std::vector<std::shared_ptr<MaterialGroup>> &mg = r.getMaterialGroups();
	for (unsigned int g = 0; g < mg.size(); ++g) {
		if (mg[g]->getNumberOfPrimitives() > 0) {
			try {
				optix::Geometry geom = m_Context->createGeometry();
				geom->setPrimitiveCount(mg[g]->getNumberOfPrimitives());
				geom->setBoundingBoxProgram(m_BoundingBox);
				geom->setIntersectionProgram(m_GeomIntersect);
				geom["vertex_buffer"]->setBuffer(m_BufferLib->getBuffer(v->getBufferID(0),size));
				for (unsigned int b = 1; b < VertexData::MaxAttribs; ++b) {
					if (m_VertexAttributes[b] && v->getBufferID(b))
						geom[VertexData::Syntax[b]]->setBuffer(m_BufferLib->getBuffer(v->getBufferID(b),size));		
				}

				geom["index_buffer"]->setBuffer(
					m_BufferLib->getIndexBuffer(mg[g]->getIndexData()->getBufferID(), 
												mg[g]->getIndexData()->getIndexSize()));

				m_GeomInstances.push_back(m_Context->createGeometryInstance());
				m_GeomInstances[m_GeomInstances.size()-1]->setMaterialCount(1);
				m_MaterialLib->applyMaterial(m_GeomInstances[m_GeomInstances.size()-1], 
					materialMap[mg[g]->getMaterialName()]);
				m_GeomInstances[m_GeomInstances.size()-1]->setGeometry(geom);
				
			}
			catch ( optix::Exception& e ) {
				NAU_THROW("Optix Error: Adding scene object %s, material %s(creating buffers from VBOs) [%s]",
						r.getName().c_str(), mg[g]->getMaterialName().c_str(), e.getErrorString().c_str()); 
			}
		}
	}
}


void
OptixGeometry::buildGeometryGroup() {

	m_GeomGroup = m_Context->createGeometryGroup();
	try {
		m_GeomGroup->setChildCount((unsigned int)m_GeomInstances.size());
		for (unsigned int i = 0; i < m_GeomInstances.size(); ++i) {
			m_GeomGroup->setChild(i,m_GeomInstances[i]);
		}
		optix::Acceleration accel = m_Context->createAcceleration("Bvh", "Bvh");
		/*accel->setProperty( "vertex_buffer_name", "vertex_buffer");
		accel->setProperty( "index_buffer_name", "index_buffer");
		accel->setProperty( "vertex_buffer_stride", "4");*/
		m_GeomGroup->setAcceleration(accel);

	}
	catch ( optix::Exception& e ) {
		NAU_THROW("Optix Error: Building Geometry Group [%s]",
											e.getErrorString().c_str()); 
	}
}


optix::GeometryGroup &
OptixGeometry::getGeometryGroup() {

	return m_GeomGroup;
}