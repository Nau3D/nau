#include <nau/scene/sceneobject.h>

#include <nau.h>
#include <nau/geometry/boundingbox.h>
#include <nau/math/transformfactory.h>

using namespace nau::scene;
using namespace nau::render;
using namespace nau::geometry;
using namespace nau::math;

unsigned int SceneObject::Counter = 0;


bool
SceneObject::Init() {

	// VEC4
	Attribs.add(Attribute(SCALE, "SCALE", Enums::DataType::VEC4, false, new vec4(1.0f, 1.0f, 1.0f, 1.0f)));
	Attribs.add(Attribute(TRANSLATE, "TRANSLATE", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, -1.0f, 0.0f)));
	Attribs.add(Attribute(ROTATE, "ROTATE", Enums::DataType::VEC4, true, new vec4(0.0f, 0.0f, 1.0f, 0.0f)));

	NAU->registerAttributes("SCENE", &Attribs);

	return true;
}

AttribSet SceneObject::Attribs;
bool SceneObject::Inited = Init();

void
SceneObject::setPropf4(Float4Property prop, vec4& aVec) {

	ITransform *tis = TransformFactory::create("SimpleTransform");

	switch (prop) {
	case SCALE:
		tis->scale(aVec.x, aVec.y, aVec.z);
		transform(tis);
		break;
	case ROTATE:
		tis->rotate(aVec.w, aVec.x, aVec.y, aVec.z);
		transform(tis);
		break;
	case TRANSLATE:
		tis->translate(aVec.x, aVec.y, aVec.z);
		transform(tis);
		break;
	}
	m_Float4Props[prop] = aVec;

}
void
SceneObject::ResetCounter(void) {
	Counter = 1;
}


SceneObject::SceneObject(void) :
	m_Id (0),
	m_Name ("default"),
	m_Renderable (0),
	m_BoundingVolume (0),
	m_StaticCondition(true)
{
	m_Transform = TransformFactory::create("SimpleTransform");
	m_ResultTransform = TransformFactory::create("SimpleTransform");
	m_GlobalTransform = TransformFactory::create("SimpleTransform");
	m_Id = SceneObject::Counter++;
//	m_BoundingVolume = new BoundingBox;
}


SceneObject::~SceneObject(void) {

	delete m_BoundingVolume;
	if (m_Transform)
		delete m_Transform;
	delete m_GlobalTransform;
	delete m_ResultTransform;
}


bool
SceneObject::isStatic() {

	return m_StaticCondition;
}


// Objects can be static or dynamic
// true means static, false means dynamic
void 
SceneObject::setStaticCondition(bool aCondition) {

	m_StaticCondition = aCondition;
}


int 
SceneObject::getId ()
{
	return m_Id;
}


void 
SceneObject::setId (int id)
{
	this->m_Id = id;
}


std::string& 
SceneObject::getName ()
{
	return m_Name;
}


void 
SceneObject::setName (const std::string &name)
{
	this->m_Name = name;
}


const nau::geometry::IBoundingVolume*
SceneObject::getBoundingVolume()
{
	if (0 == m_BoundingVolume) {
		calculateBoundingVolume();
		if (m_Transform)
			m_BoundingVolume->setTransform (*m_ResultTransform);
	}
//	m_BoundingVolume->setTransform (*m_ResultTransform);
	return (m_BoundingVolume);
}


void
SceneObject::setBoundingVolume (IBoundingVolume *b)
{
	if (0 != m_BoundingVolume) {
		delete (m_BoundingVolume);
	}
	m_BoundingVolume = b;
}


const ITransform& 
SceneObject::getTransform ()
{
	return (*m_ResultTransform);
}


void 
SceneObject::burnTransform(void)
{
	const mat4 &transformationMatrix = m_ResultTransform->getMat44();

	ITransform *aux = m_ResultTransform->clone();
	aux->invert();
	aux->transpose();
	const mat4 &normalMatrix = aux->getMat44();

	VertexData &vertexData = m_Renderable->getVertexData();

	std::vector<VertexData::Attr> &vertices = vertexData.getDataOf(VertexData::getAttribIndex("position"));
	std::vector<VertexData::Attr> &normals = vertexData.getDataOf(VertexData::getAttribIndex("normal"));
	std::vector<VertexData::Attr>::iterator verticesIter, normalIter;


	if (normals.size() > 0) {
		normalIter = normals.begin();
		for (; normalIter != normals.end(); ++normalIter) {
			normalMatrix.transform3(*normalIter);
			normalIter->normalize();
		}
	}
	verticesIter = vertices.begin();
	for (; verticesIter != vertices.end(); ++verticesIter) {
		transformationMatrix.transform(*verticesIter);
	}

	//Reset transform to the identity
	mat4 &thisMatrixShouldNotExist = const_cast<mat4 &> (transformationMatrix); 
	m_ResultTransform->setIdentity();
	m_Transform->setIdentity();
	m_GlobalTransform->setIdentity();

	//Recalculate bounding box
	calculateBoundingVolume();
}


void 
SceneObject::setTransform (ITransform *t)
{
	//if (0 != this->m_Transform){ 
	//	delete m_Transform; 
	//}
	m_Transform->clone(t);
	m_ResultTransform->clone(m_GlobalTransform);
	m_ResultTransform->compose(*m_Transform);

	if (m_BoundingVolume)
		m_BoundingVolume->setTransform(*m_ResultTransform);
}


void
SceneObject::transform(ITransform *t)
{
	//if (0 != this->m_Transform){ 
	//	delete m_Transform; 
	//}
	m_Transform->compose(*t);
	m_ResultTransform->clone(m_GlobalTransform);
	m_ResultTransform->compose(*m_Transform);

	if (m_BoundingVolume)
		m_BoundingVolume->setTransform(*m_ResultTransform);
}


void 
SceneObject::updateGlobalTransform(ITransform *t)
{
	m_GlobalTransform->clone(t);
	m_ResultTransform->clone(m_GlobalTransform);
	m_ResultTransform->compose(*m_Transform);
	if (m_BoundingVolume)
		m_BoundingVolume->setTransform(*m_ResultTransform);
}


IRenderable& 
SceneObject::getRenderable (void)
{
	return (*m_Renderable);
}


IRenderable* 
SceneObject::_getRenderablePtr (void)
{
	return m_Renderable;
}


void 
SceneObject::setRenderable (nau::render::IRenderable *renderable)
{
	m_Renderable = renderable;
}


std::string 
SceneObject::getType (void)
{
	return "SimpleObject";
}


void
SceneObject::calculateBoundingVolume (void)
{
	if (0 != m_BoundingVolume) {
		delete m_BoundingVolume;
	}

	m_BoundingVolume = new BoundingBox; /***MARK***/

	m_BoundingVolume->
		calculate (m_Renderable->getVertexData().getDataOf (VertexData::getAttribIndex("position")));

	m_BoundingVolume->setTransform(*m_ResultTransform);
}


void 
SceneObject::writeSpecificData (std::fstream &f)
{
	return;
}


void 
SceneObject::readSpecificData (std::fstream &f)
{
	return;
}


nau::math::ITransform*
SceneObject::_getTransformPtr (void)
{
	return m_ResultTransform;
}


void SceneObject::unitize(float min, float max) {
	
	m_Renderable->unitize(min,max);
	m_BoundingVolume->
		calculate(m_Renderable->getVertexData().getDataOf(VertexData::getAttribIndex("position")));
}


void 
SceneObject::prepareTriangleIDs(bool ids) 
{
	if (ids)
		m_Renderable->prepareTriangleIDs(m_Id);
}