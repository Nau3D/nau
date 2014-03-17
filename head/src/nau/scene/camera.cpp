#define _USE_MATH_DEFINES
#include <cmath>
#include <nau/slogger.h>


#include <nau/scene/camera.h>
#include <nau/math/utils.h>
#include <nau/math/simpletransform.h>
#include <nau/math/mat4.h>
#include <nau/geometry/boundingbox.h>
#include <nau/geometry/mesh.h>
#include <nau/material/materialgroup.h>
#include <nau/render/irenderer.h>
#include <nau/event/eventFactory.h> 
#include <nau/event/cameraMotion.h>
#include <nau/event/cameraOrientation.h>
#include <nau.h>


using namespace nau::scene;
using namespace nau::math;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;

bool
Camera::Init() {

	// VEC4
	Attribs.add(Attribute(POSITION, "POSITION",Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 0.0f, 1.0f)));
	Attribs.add(Attribute(VIEW_VEC, "VIEW", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, -1.0f, 0.0f)));
	Attribs.add(Attribute(NORMALIZED_VIEW_VEC, "NORMALIZED_VIEW_VEC", Enums::DataType::VEC4, true,new vec4(0.0f, 0.0f, -1.0f, 0.0f)));
	Attribs.add(Attribute(UP_VEC, "UP", Enums::DataType::VEC4, false, new vec4(0.0f, 1.0f, 0.0f, 0.0f)));
	Attribs.add(Attribute(NORMALIZED_UP_VEC, "NORMALIZED_UP" ,Enums::DataType::VEC4, true, new vec4(0.0f, 1.0f, 0.0f, 0.0f)));
	Attribs.add(Attribute(NORMALIZED_RIGHT_VEC, "NORMALIZED_RIGHT" ,Enums::DataType::VEC4, true, new vec4(1.0f, 0.0f, 0.0f, 0.0f)));
	Attribs.add(Attribute(LOOK_AT_POINT, "LOOK_AT_POINT" ,Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, -1.0f, 1.0f)));
	// MAT4
	Attribs.add(Attribute(VIEW_MATRIX, "VIEW_MATRIX",Enums::DataType::MAT4, true));
	Attribs.add(Attribute(PROJECTION_MATRIX, "PROJECTION_MATRIX",Enums::DataType::MAT4, true));
	Attribs.add(Attribute(VIEW_INVERSE_MATRIX, "VIEW_INVERSE_MATRIX",Enums::DataType::MAT4, true));
	Attribs.add(Attribute(PROJECTION_VIEW_MATRIX, "PROJECTION_VIEW_MATRIX",Enums::DataType::MAT4, true));
	Attribs.add(Attribute(TS05_PVM_MATRIX, "TS05_PVM_MATRIX",Enums::DataType::MAT4, true));
	// FLOAT
	Attribs.add(Attribute(FOV, "FOV",Enums::DataType::FLOAT, false, new float(60.0f)));
	Attribs.add(Attribute(NEARP, "NEARP",Enums::DataType::FLOAT, false, new float(1.0f)));
	Attribs.add(Attribute(FARP, "FARP",Enums::DataType::FLOAT, false, new float(10000.0f)));
	Attribs.add(Attribute(LEFT, "LEFT",Enums::DataType::FLOAT, false, new float(-1.0f)));
	Attribs.add(Attribute(RIGHT, "RIGHT",Enums::DataType::FLOAT, false, new float(1.0f)));
	Attribs.add(Attribute(TOP, "TOP",Enums::DataType::FLOAT, false, new float(1.0f)));
	Attribs.add(Attribute(BOTTOM, "BOTTOM",Enums::DataType::FLOAT, false, new float(-1.0f)));
	Attribs.add(Attribute(ZX_ANGLE, "ZX_ANGLE", Enums::DataType::FLOAT, false, new float((float)M_PI)));
	Attribs.add(Attribute(ELEVATION_ANGLE, "ELEVATION_ANGLE", Enums::DataType::FLOAT, false, new float(0.0f)));
	// ENUM
	Attribs.add(Attribute(PROJECTION_TYPE, "TYPE", Enums::DataType::ENUM, false, new int(PERSPECTIVE)));
	Attribs.listAdd("TYPE", "PERSPECTIVE", PERSPECTIVE);
	Attribs.listAdd("TYPE", "ORTHO", ORTHO);
	return true;
}


AttribSet Camera::Attribs;
bool Camera::Inited = Init();

void
Camera::setDefault()
{
	Attribs.initAttribInstanceFloatArray(m_FloatProps);
	Attribs.initAttribInstanceVec4Array(m_Float4Props);
	Attribs.initAttribInstanceEnumArray(m_EnumProps);
}


Camera::Camera (const std::string &name) :
	SceneObject(),

	m_IsDynamic (false),
	m_pViewport (0),
	m_LookAt(false),
	//m_LookAtPoint(0.0f, 0.0f, 0.0f),
	m_PositionOffset (0.0f)
	//m_IsOrtho (false)
{
	setDefault();
	m_Id = 0;
	m_Name = name;
	m_pViewport = NAU->getDefaultViewport();

	//m_Float4Props[POSITION].set(0.0f, 0.0f, 0.0f, 1.0f);
	//m_Float4Props[VIEW_VEC].set(0.0f, 0.0f, -1.0f, 0.0f);
	//m_Float4Props[NORMALIZED_VIEW_VEC].set(0.0f, 0.0f, -1.0f, 0.0f);

	//m_Float4Props[NORMALIZED_RIGHT_VEC].set(1.0f, 0.0f, 0.0f, 0.0f);

	//m_Float4Props[UP_VEC].set(0.0f, 1.0f, 0.0f, 0.0f);
	//m_Float4Props[NORMALIZED_UP_VEC].set(0.0f, 1.0f, 0.0f, 0.0f);

	//m_Float4Props[LOOK_AT_POINT].set(0.0f, 0.0f, -1.0f, 1.0f);

	//m_FloatProps[FOV] = 60.0f;
	//m_FloatProps[TOP] = 1.0f;
	//m_FloatProps[BOTTOM] = -1.0f;
	//m_FloatProps[LEFT] = -1.0f;
	//m_FloatProps[RIGHT] = 1.0f;
	//m_FloatProps[NEARP] = 1.0f;
	//m_FloatProps[FARP] = 10000.0f;
	//m_FloatProps[ELEVATION_ANGLE] = 0.0f;
	//m_FloatProps[ZX_ANGLE] = (float)M_PI;

	buildViewMatrix();
	buildViewMatrixInverse();

	m_StaticCondition = false;

	m_BoundingVolume = new BoundingBox;
	m_Transform = new SimpleTransform;
	setVectorsFromSpherical();

	// Adding a Mesh with the frustum lines
	Mesh *renderable =  (Mesh *)RESOURCEMANAGER->createRenderable("Mesh", m_Name, "Camera");
	renderable->setDrawingPrimitive(nau::render::IRenderer::LINES);
	std::vector<VertexData::Attr> *vertices = new std::vector<VertexData::Attr>(8);
	VertexData &vertexData = renderable->getVertexData();
	vertexData.setDataFor (VertexData::getAttribIndex("position"), vertices);

	MaterialGroup *aMaterialGroup = new MaterialGroup;
	
	std::vector<unsigned int> *indices = new std::vector<unsigned int>(16);
	indices->at (0) = Camera::TOP_LEFT_NEAR;		indices->at (1) = Camera::TOP_LEFT_FAR;
	indices->at (2) = Camera::TOP_RIGHT_NEAR;		indices->at (3) = Camera::TOP_RIGHT_FAR;
	indices->at (4) = Camera::BOTTOM_RIGHT_NEAR;	indices->at (5) = Camera::BOTTOM_RIGHT_FAR;
	indices->at (6) = Camera::BOTTOM_LEFT_NEAR;		indices->at (7) = Camera::BOTTOM_LEFT_FAR;

	indices->at (8) = Camera::TOP_LEFT_FAR;			indices->at (9) = Camera::TOP_RIGHT_FAR;
	indices->at (10) = Camera::TOP_RIGHT_FAR;		indices->at (11) = Camera::BOTTOM_RIGHT_FAR;
	indices->at (12) = Camera::BOTTOM_RIGHT_FAR;	indices->at (13) = Camera::BOTTOM_LEFT_FAR;
	indices->at (14) = Camera::BOTTOM_LEFT_FAR;		indices->at (15) = Camera::TOP_LEFT_FAR;

	aMaterialGroup->setIndexList (indices);
	aMaterialGroup->setParent (renderable);
//	aMaterialGroup->setMaterialName("__Black");
	aMaterialGroup->setMaterialName("__Emission White");

	renderable->addMaterialGroup (aMaterialGroup);
	m_Transform = & m_Mat4Props[VIEW_INVERSE_MATRIX];
	setRenderable (renderable);

	aMaterialGroup = new MaterialGroup;
	indices = new std::vector<unsigned int>(8);
	indices->at (0) = Camera::TOP_LEFT_NEAR;		indices->at (1) = Camera::TOP_RIGHT_NEAR;
	indices->at (2) = Camera::TOP_RIGHT_NEAR;		indices->at (3) = Camera::BOTTOM_RIGHT_NEAR;
	indices->at (4) = Camera::BOTTOM_RIGHT_NEAR;	indices->at (5) = Camera::BOTTOM_LEFT_NEAR;
	indices->at (6) = Camera::BOTTOM_LEFT_NEAR;		indices->at (7) = Camera::TOP_LEFT_NEAR;

	aMaterialGroup->setIndexList (indices);
	aMaterialGroup->setParent (renderable);
//	aMaterialGroup->setMaterialName("__Black");
	aMaterialGroup->setMaterialName("__Emission Red");

	renderable->addMaterialGroup (aMaterialGroup);
	setRenderable (renderable);

	IScene *s = RENDERMANAGER->createScene(name, "SceneAux");
	s->add(this);

	EVENTMANAGER->addListener("VIEWPORT_CHANGED", this);
}


Camera::~Camera (void)
{
	m_Transform = NULL;
	m_pViewport = NULL;
}


void
Camera::setOrtho(float left, float right, float bottom, float top, float near, float far)
{
	m_EnumProps[PROJECTION_TYPE] = ORTHO;
	m_FloatProps[LEFT] = left;
	m_FloatProps[RIGHT] = right;
	m_FloatProps[BOTTOM] = bottom;
	m_FloatProps[TOP] = top;
	m_FloatProps[NEARP] = near;
	m_FloatProps[FARP] = far;

	buildProjectionMatrix();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


void 
Camera::setPerspective (float fov, float near, float far)
{
	m_EnumProps[PROJECTION_TYPE] = PERSPECTIVE;
	m_FloatProps[FOV] = fov;
	m_FloatProps[NEARP] = near;
	m_FloatProps[FARP] = far;

	buildProjectionMatrix();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


// to be called when viewport changes
void 
Camera::updateProjection ()
{
	buildProjectionMatrix();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


void Camera::setProp(Mat4Property prop, mat4 &mat) 
{
	m_Mat4Props[prop].setMat44(mat);
}


void 
Camera::setProp(FloatProperty prop, float f) 
{
	m_FloatProps[prop] = f;
	vec3 v;

	switch(prop) {

		case ZX_ANGLE:
		case ELEVATION_ANGLE:
			setVectorsFromSpherical();
			break;
		default:
			buildProjectionMatrix();
			buildProjectionViewMatrix();
			buildTS05PVMMatrix();
	}
}


void
Camera::setProp(Float4Property prop, float x, float y, float z, float w)
{
	vec4 v;
	vec2 v2;

	v.set(x,y,z,w);

	switch(prop) {

		case POSITION:
			v.w = 1;	
			m_Float4Props[POSITION].set(v);
			m_Float4Props[LOOK_AT_POINT] = m_Float4Props[POSITION];
			m_Float4Props[LOOK_AT_POINT] += m_Float4Props[NORMALIZED_VIEW_VEC];
			break;

		case VIEW_VEC:
		case NORMALIZED_VIEW_VEC:
			v.w = 0.0f;
			v2 = Spherical::toSpherical(x,y,z);
			m_FloatProps[ZX_ANGLE] = v2.x;
			m_FloatProps[ELEVATION_ANGLE] = v2.y;
			m_Float4Props[VIEW_VEC].set(v);
			m_Float4Props[NORMALIZED_VIEW_VEC].set(v);
			m_Float4Props[NORMALIZED_VIEW_VEC].normalize();
			m_Float4Props[NORMALIZED_RIGHT_VEC] = m_Float4Props[NORMALIZED_VIEW_VEC].cross(m_Float4Props[NORMALIZED_UP_VEC]);
			m_Float4Props[NORMALIZED_RIGHT_VEC].normalize();
			m_Float4Props[NORMALIZED_UP_VEC] = m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[NORMALIZED_VIEW_VEC]);
			m_Float4Props[UP_VEC].set(m_Float4Props[NORMALIZED_UP_VEC]);
			m_Float4Props[LOOK_AT_POINT] = m_Float4Props[POSITION];
			m_Float4Props[LOOK_AT_POINT] += m_Float4Props[NORMALIZED_VIEW_VEC];
			break;

		case UP_VEC:
			v.w = 0.0f;
			m_Float4Props[NORMALIZED_RIGHT_VEC].set(m_Float4Props[VIEW_VEC].cross(v));
			m_Float4Props[NORMALIZED_RIGHT_VEC].normalize();
			m_Float4Props[NORMALIZED_UP_VEC].set(m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[NORMALIZED_VIEW_VEC]));
			m_Float4Props[UP_VEC] = m_Float4Props[NORMALIZED_UP_VEC];
			break;

		case LOOK_AT_POINT:
			v.w = 1.0f;
			m_Float4Props[LOOK_AT_POINT].set(v);
			m_Float4Props[VIEW_VEC] = m_Float4Props[LOOK_AT_POINT];
			m_Float4Props[VIEW_VEC] -= m_Float4Props[POSITION];
			m_Float4Props[VIEW_VEC].w = 0.0f;
			m_Float4Props[NORMALIZED_VIEW_VEC].set(m_Float4Props[VIEW_VEC]);
			m_Float4Props[NORMALIZED_VIEW_VEC].normalize();
			v2 = Spherical::toSpherical(m_Float4Props[NORMALIZED_VIEW_VEC].x,m_Float4Props[NORMALIZED_VIEW_VEC].y,m_Float4Props[NORMALIZED_VIEW_VEC].z);
			m_FloatProps[ZX_ANGLE] = v2.x;
			m_FloatProps[ELEVATION_ANGLE] = v2.y;

			m_Float4Props[NORMALIZED_RIGHT_VEC] = m_Float4Props[NORMALIZED_VIEW_VEC].cross(m_Float4Props[NORMALIZED_UP_VEC]);
			m_Float4Props[NORMALIZED_RIGHT_VEC].normalize();
			m_Float4Props[NORMALIZED_UP_VEC] = m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[NORMALIZED_VIEW_VEC]);
			break;
	}
	buildViewMatrix();
	buildViewMatrixInverse();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


void 
Camera::setProp(EnumProperty prop, int value) 
{
	m_EnumProps[prop] = value;

	buildProjectionMatrix();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


void 
Camera::setProp(int prop, Enums::DataType type, void *value) {

	vec4 *v;
	switch (type) {

		case Enums::FLOAT:
			if (prop < COUNT_FLOATPROPERTY)
				setProp((FloatProperty)prop, *(float *)value);
			else
				m_FloatProps[prop] = *(float *)value;
			break;
		case Enums::VEC4:
			v = (vec4 *)value;
			if (prop < COUNT_FLOAT4PROPERTY)
				setProp((Float4Property)prop, v->x, v->y, v->z, v->w);
			else
				m_Float4Props[prop].set((vec4 *)value);
			break;
		case Enums::INT:
			m_IntProps[prop] = *(int *)value;
			break;
	}
}


void *
Camera::getProp(int prop, Enums::DataType type) {

	switch (type) {

	case Enums::FLOAT:
		assert(m_FloatProps.count(prop) > 0);
		return(&(m_FloatProps[prop]));
		break;
	case Enums::VEC4:
		assert(m_Float4Props.count(prop) > 0);
		return(&(m_Float4Props[prop]));
		break;
	case Enums::INT:
		assert(m_IntProps.count(prop) > 0);
		return(&(m_IntProps[prop]));
		break;
	}
	return NULL;
}


const vec4&
Camera::getPropf4(Float4Property prop) 
{
	return m_Float4Props[prop];
}


float 
Camera::getPropf(FloatProperty prop)
{
	return m_FloatProps[prop];
}


const mat4&
Camera::getPropm4(Mat4Property prop)
{
	return m_Mat4Props[prop].getMat44();
}


int
Camera::getPrope(EnumProperty prop) 
{
	return m_EnumProps[prop];
}



IRenderable& 
Camera::getRenderable (void)
{
	vec3 frustumPoints[8];

	/// MARK - This can be done only when modifying the camera parameters
	std::vector<VertexData::Attr> *vertices = new std::vector<VertexData::Attr>(8);

	if (m_EnumProps[PROJECTION_TYPE] == ORTHO) {

		vertices->at (TOP_LEFT_NEAR).set     (m_FloatProps[LEFT],  m_FloatProps[TOP],   -m_FloatProps[NEARP]);
		vertices->at (TOP_RIGHT_NEAR).set    (m_FloatProps[RIGHT], m_FloatProps[TOP],   -m_FloatProps[NEARP]);
		vertices->at (BOTTOM_RIGHT_NEAR).set (m_FloatProps[RIGHT], m_FloatProps[BOTTOM],-m_FloatProps[NEARP]);
		vertices->at (BOTTOM_LEFT_NEAR).set  (m_FloatProps[LEFT],  m_FloatProps[BOTTOM],-m_FloatProps[NEARP]);
		vertices->at (TOP_LEFT_FAR).set      (m_FloatProps[LEFT],  m_FloatProps[TOP],   -m_FloatProps[FARP]);
		vertices->at (TOP_RIGHT_FAR).set     (m_FloatProps[RIGHT], m_FloatProps[TOP],   -m_FloatProps[FARP]);
		vertices->at (BOTTOM_RIGHT_FAR).set  (m_FloatProps[RIGHT], m_FloatProps[BOTTOM],-m_FloatProps[FARP]);
		vertices->at (BOTTOM_LEFT_FAR).set   (m_FloatProps[LEFT],  m_FloatProps[BOTTOM],-m_FloatProps[FARP]);
	}
	else {
		float hh = tan(DegToRad(m_FloatProps[FOV])/2.0);
		float hw = hh * m_pViewport->getPropf(Viewport::RATIO);

		vertices->at (TOP_LEFT_NEAR).set (-hw*m_FloatProps[NEARP], hh*m_FloatProps[NEARP], -m_FloatProps[NEARP]);
		vertices->at (TOP_RIGHT_NEAR).set (hw*m_FloatProps[NEARP], hh*m_FloatProps[NEARP], -m_FloatProps[NEARP]);
		vertices->at (BOTTOM_RIGHT_NEAR).set (hw*m_FloatProps[NEARP], -hh*m_FloatProps[NEARP], -m_FloatProps[NEARP]);
		vertices->at (BOTTOM_LEFT_NEAR).set (-hw*m_FloatProps[NEARP], -hh*m_FloatProps[NEARP], -m_FloatProps[NEARP]);
		vertices->at (TOP_LEFT_FAR).set (-hw*m_FloatProps[FARP], hh * m_FloatProps[FARP], -m_FloatProps[FARP]);
		vertices->at (TOP_RIGHT_FAR).set (hw*m_FloatProps[FARP], hh*m_FloatProps[FARP], -m_FloatProps[FARP]);
		vertices->at (BOTTOM_RIGHT_FAR).set (hw*m_FloatProps[FARP], -hh*m_FloatProps[FARP], -m_FloatProps[FARP]);
		vertices->at (BOTTOM_LEFT_FAR).set (-hw*m_FloatProps[FARP], -hh*m_FloatProps[FARP], -m_FloatProps[FARP]);	
	}

	VertexData &vertexData = m_Renderable->getVertexData();
	vertexData.setDataFor (VertexData::getAttribIndex("position"), vertices);

	std::vector<VertexData::Attr> *normals = new std::vector<VertexData::Attr>(8);
	for (int i = 0; i < 8 ; ++i) 
		normals->at(i).set(0.0f, 0.0f, 0.0f);
	vertexData.setDataFor (VertexData::getAttribIndex("normal"), normals);

	buildViewMatrixInverse();
	m_ResultTransform->clone(m_GlobalTransform);
	m_ResultTransform->compose(*m_Transform);
	return (*m_Renderable);
}


const IBoundingVolume*
Camera::getBoundingVolume ()
{
	calculateBoundingVolume();
	m_BoundingVolume->setTransform (m_Mat4Props[VIEW_INVERSE_MATRIX]);
	return (m_BoundingVolume);
}


void
Camera::setCamera (vec3 position, vec3 view, vec3 up)
{
	if (m_IsDynamic) {
		m_Float4Props[POSITION].set(position.x, position.y + 0.85f, position.z, 1.0f);
	} else {
		m_Float4Props[POSITION].set(position.x, position.y, position.z, 1.0f);
	}

	m_Float4Props[VIEW_VEC].set(view.x, view.y, view.z, 0.0f);
	m_Float4Props[NORMALIZED_VIEW_VEC].set(view.x, view.y, view.z, 0.0f);
	m_Float4Props[NORMALIZED_VIEW_VEC].normalize();

	vec2 vs = Spherical::toSpherical(view.x, view.y, view.z);

	m_Float4Props[UP_VEC].set(up.x, up.y, up.z, 0.0f);

	m_Float4Props[NORMALIZED_RIGHT_VEC].set(m_Float4Props[VIEW_VEC].cross(m_Float4Props[UP_VEC]));
	m_Float4Props[NORMALIZED_RIGHT_VEC].normalize();

	m_Float4Props[NORMALIZED_UP_VEC].set(m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[NORMALIZED_VIEW_VEC]));
	m_Float4Props[NORMALIZED_UP_VEC].normalize();
	m_Float4Props[UP_VEC].set(m_Float4Props[NORMALIZED_UP_VEC]);

	vec4 v4 = m_Float4Props[POSITION];
	v4 += m_Float4Props[NORMALIZED_VIEW_VEC];
	m_Float4Props[LOOK_AT_POINT].set(v4.x, v4.y, v4.z, 1.0f);

	buildViewMatrix();
	buildViewMatrixInverse();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();

}


void 
Camera::buildViewMatrixInverse(void) {

	// This is a simpler inverse because the view matrix has a specific format
	//mat4& tmp = const_cast<mat4&>(m_ViewMatrix->getMat44());
	//mat4& tmp2 = const_cast<mat4&>(m_ViewMatrixInverse->getMat44());
	mat4& tmp = const_cast<mat4&>(m_Mat4Props[VIEW_MATRIX].getMat44());
	mat4& tmp2 = const_cast<mat4&>(m_Mat4Props[VIEW_INVERSE_MATRIX].getMat44());

	int i,j;
	float aux;

	for (i = 0; i < 3; i ++) {
		for (j = 0; j <3 ; j ++) 
			tmp2.set(i,j,tmp.at(j,i));
	}
	for (i = 0; i < 3 ; i ++)
		tmp2.set(i,3,0.0f);
	tmp2.set(3,3,1.0f);

	for (i = 0 ; i < 3; i++) {
		aux = 0.0f;
		for (j = 0; j < 3 ; j++)
			aux += tmp.at(i,j) * tmp.at(3,j);
		tmp2.set(3,i,-aux);
	}
}


void 
Camera::setVectorsFromSpherical() 
{
	// maybe only do this if there is no look at point?
	m_FloatProps[ELEVATION_ANGLE] = Spherical::capBeta(m_FloatProps[ELEVATION_ANGLE]);

	// View vector = -Z from camera
	vec3 v;
	v = Spherical::toCartesian(m_FloatProps[ZX_ANGLE], m_FloatProps[ELEVATION_ANGLE]);
	m_Float4Props[NORMALIZED_VIEW_VEC].set( v.x, v.y, v.z, 0.0f);
	m_Float4Props[VIEW_VEC].set(v.x, v.y, v.z, 0.0f);

	v = Spherical::getRightVector(m_FloatProps[ZX_ANGLE], m_FloatProps[ELEVATION_ANGLE]);
	m_Float4Props[NORMALIZED_RIGHT_VEC].set(v.x, v.y, v.z, 0.0f);

	vec4 v4 = m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[NORMALIZED_VIEW_VEC]);
	m_Float4Props[NORMALIZED_UP_VEC].set(v4.x, v4.y, v4.z, 0.0f);
	m_Float4Props[UP_VEC].set(v4.x, v4.y, v4.z, 0.0f);

	v4 = m_Float4Props[POSITION];
	v4 += m_Float4Props[NORMALIZED_VIEW_VEC];
	m_Float4Props[LOOK_AT_POINT].set(v4.x, v4.y, v4.z, 1.0f);

	buildViewMatrix();
	buildViewMatrixInverse();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


void 
Camera::buildProjectionMatrix() {

	float aspect = m_pViewport->getPropf(Viewport::RATIO);
	float f = 1.0f / tan (DegToRad(m_FloatProps[FOV] * 0.5f));

	m_Mat4Props[PROJECTION_MATRIX].setIdentity();

	mat4& projection = const_cast<mat4&>(m_Mat4Props[PROJECTION_MATRIX].getMat44()); 

	if (m_EnumProps[PROJECTION_TYPE] == PERSPECTIVE) {
		projection.set (0, 0, f / aspect);
		projection.set (1, 1, f);
		projection.set (2, 2, (m_FloatProps[FARP] + m_FloatProps[NEARP]) / (m_FloatProps[NEARP] - m_FloatProps[FARP]));
		projection.set (3, 2, (2.0f * m_FloatProps[FARP] * m_FloatProps[NEARP]) / (m_FloatProps[NEARP] - m_FloatProps[FARP]));
		projection.set (2, 3, -1.0f);
		projection.set (3, 3, 0.0f);
	}
	else {
		projection.set (0, 0, 2 / (m_FloatProps[RIGHT] - m_FloatProps[LEFT]));
		projection.set (1, 1, 2 / (m_FloatProps[TOP] - m_FloatProps[BOTTOM]));
		projection.set (2, 2, -2 / (m_FloatProps[FARP] - m_FloatProps[NEARP]));
		projection.set (3, 0, -(m_FloatProps[RIGHT] + m_FloatProps[LEFT]) / (m_FloatProps[RIGHT] - m_FloatProps[LEFT]));
		projection.set (3, 1, -(m_FloatProps[TOP] + m_FloatProps[BOTTOM]) / (m_FloatProps[TOP] - m_FloatProps[BOTTOM]));
		projection.set (3, 2, -(m_FloatProps[FARP] + m_FloatProps[NEARP]) / (m_FloatProps[FARP] - m_FloatProps[NEARP]));
	}

	if (this->m_Renderable)
		this->m_Renderable->resetCompilationFlags();

}
	

void
Camera::setViewport (Viewport* aViewport)
{
	m_pViewport = aViewport;

	buildProjectionMatrix();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


Viewport *
Camera::getViewport (void)
{
	return (m_pViewport);
}



void
Camera::buildTS05PVMMatrix(void) 
{
	m_Mat4Props[TS05_PVM_MATRIX].setIdentity();

	m_Mat4Props[TS05_PVM_MATRIX].translate (0.5f, 0.5f, 0.5f);
	m_Mat4Props[TS05_PVM_MATRIX].scale (0.5f);

	m_Mat4Props[TS05_PVM_MATRIX].compose (m_Mat4Props[PROJECTION_VIEW_MATRIX]);
}

void
Camera::buildProjectionViewMatrix (void)
{
	m_Mat4Props[PROJECTION_VIEW_MATRIX].setIdentity();
	m_Mat4Props[PROJECTION_VIEW_MATRIX].compose(m_Mat4Props[PROJECTION_MATRIX]);
	m_Mat4Props[PROJECTION_VIEW_MATRIX].compose(m_Mat4Props[VIEW_MATRIX]);
}

void
Camera::buildViewMatrix (void)
{
	vec4 s,u,v;
	u = m_Float4Props[NORMALIZED_UP_VEC];
	s = m_Float4Props[NORMALIZED_RIGHT_VEC];
	v = m_Float4Props[NORMALIZED_VIEW_VEC];

	mat4& viewMatrix = const_cast<mat4&>(m_Mat4Props[VIEW_MATRIX].getMat44()); 

	viewMatrix.setIdentity();

	viewMatrix.set (0, 0, s.x);
	viewMatrix.set (1, 0, s.y);
	viewMatrix.set (2, 0, s.z);

	viewMatrix.set (0, 1, u.x);
	viewMatrix.set (1, 1, u.y);
	viewMatrix.set (2, 1, u.z);

	viewMatrix.set (0, 2, -v.x);
	viewMatrix.set (1, 2, -v.y);
	viewMatrix.set (2, 2, -v.z);

	if (m_IsDynamic) {
		vec3 p = m_Transform->getTranslation();
		m_Mat4Props[VIEW_MATRIX].translate(-p.x, -p.y-0.85, -p.z);
	}
	else {
		vec4 p = m_Float4Props[POSITION];
		m_Mat4Props[VIEW_MATRIX].translate(-p.x, -p.y, -p.z);
	}
	if (this->m_Renderable)
		this->m_Renderable->resetCompilationFlags();

}



void
Camera::adjustMatrix(Camera* aCamera)
{
	float cNear = aCamera->getPropf(NEARP);
	float cFar = aCamera->getPropf(FARP);
	adjustMatrixPlus(cNear,cFar,aCamera);
}



void
Camera::adjustMatrixPlus(float cNear, float cFar, Camera  *aCamera)
{
	float ratio = aCamera->getViewport()->getPropf(Viewport::RATIO);
	
	float fov = aCamera->getPropf(FOV);

	float hNear = 2.0f * tan (DegToRad(fov * 0.5f)) * cNear;
	float wNear = hNear * ratio;

	float hFar = 2.0f * tan (DegToRad(fov * 0.5f)) * cFar;
	float wFar = hFar * ratio;

	vec4 rightVector = aCamera->getPropf4(NORMALIZED_RIGHT_VEC);

	vec4 upHFar (aCamera->getPropf4(NORMALIZED_UP_VEC));
	upHFar *= (hFar * 0.5f);

	vec4 upHNear (aCamera->getPropf4(NORMALIZED_UP_VEC));
	upHNear *= (hNear * 0.5f);

	vec4 rightWFar (rightVector);
	rightWFar *= (wFar * 0.5f);

	vec4 rightWNear (rightVector);
	rightWNear *= (wNear * 0.5f);

	vec4 view = aCamera->getPropf4(NORMALIZED_VIEW_VEC);
	vec4 fc = view;
	fc *= cFar;
	fc += aCamera->getPropf4(POSITION);

	vec4 points[8];

	points[0] = fc;
	points[0] += upHFar;
	points[0] -= rightWFar;

	points[1] = fc;
	points[1] += upHFar;
	points[1] += rightWFar;

	points[2] = fc;
	points[2] -= upHFar;
	points[2] -= rightWFar;

	points[3] = fc;
	points[3] -= upHFar;
	points[3] += rightWFar;

	vec4 nc = view;
	nc *= cNear;
	nc += aCamera->getPropf4(POSITION);

	points[4] = nc;
	points[4] += upHNear;
	points[4] -= rightWNear;

	points[5] = nc;
	points[5] += upHNear;
	points[5] += rightWNear;

	points[6] = nc;
	points[6] -= upHNear;
	points[6] -= rightWNear;

	points[7] = nc;
	points[7] -= upHNear;
	points[7] += rightWNear;


	//for (int i = 0; i < 8; i++) {
	//	if (points[i].y < 4.0f) {
	//		points[i].y = 4.0f;
	//	}
	//}

//	m_UpVector = aCamera->getUpVector();
	ITransform &viewMatrix = m_Mat4Props[VIEW_MATRIX];

	for (int i = 0; i < 8; i++){
		viewMatrix.getMat44().transform (points[i]);
	}



	vec3 maxPoint (-200000.0f, -200000.0f, -200000.0f); 
	vec3 minPoint (200000.0f, 200000.0f, 200000.0f);
	for (int i = 0; i < 8; i++){
		if (points[i].x > maxPoint.x) {
			maxPoint.x = points[i].x;
		}
		if (points[i].y > maxPoint.y) {
			maxPoint.y = points[i].y;
		}
		if (points[i].z > maxPoint.z) {
			maxPoint.z = points[i].z;
		}

		if (points[i].x < minPoint.x) {
			minPoint.x = points[i].x;
		}
		if (points[i].y < minPoint.y) {
			minPoint.y = points[i].y;
		}
		if (points[i].z < minPoint.z) {
			minPoint.z = points[i].z;
		}
	}

	float top = maxPoint.y;
	float bottom = minPoint.y;

	float right = maxPoint.x;
	float left = minPoint.x;

	float zNear = -maxPoint.z;
	float zFar = -minPoint.z;
	//zNear = 0.0;
	//zFar = -minPoint.z;
	//zNear = zNear - 200.0f;
	//zFar = zFar + 200.0f;

	float fe = 0.0f;

	setOrtho(left - fe, right + fe, bottom - fe, top + fe, zNear - fe, zFar + fe);
}


void 
Camera::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
	if (eventType == "VIEWPORT_CHANGED" && m_pViewport->getName() == sender)
		updateProjection();

	if (eventType == "DYNAMIC_CAMERA") {

		vec3 p = m_Transform->getTranslation();
		setProp(POSITION,p.x,p.y,p.z,1.0f);
		buildViewMatrix();

		result.set(p.x,p.y,p.z); 
		m_Event.setData(&result);
		EVENTMANAGER->notifyEvent("CAMERA_POSITION", m_Name,"", &m_Event);

	}
	if(eventType == "CAMERA_ORIENTATION"  && !m_LookAt) {
		CameraOrientation *f=(CameraOrientation *)evt->getData();
		m_FloatProps[ELEVATION_ANGLE] = f->getBeta();
		m_FloatProps[ZX_ANGLE] = f->getAlpha();
	
		setVectorsFromSpherical();
	}

	if(eventType == "CAMERA_MOTION") {
		
		CameraMotion *f=(CameraMotion *)evt->getData();
		
		float vel=f->getVelocity();

		vec4 vPos = m_Float4Props[POSITION];
		vec4 vView = m_Float4Props[NORMALIZED_VIEW_VEC];
		vec4 vRight = m_Float4Props[NORMALIZED_RIGHT_VEC];
		vec4 vUp = m_Float4Props[NORMALIZED_UP_VEC];

		if(f->getDirection()=="BACKWARD") {

			vView *= vel;
			vPos -=  vView;
			setProp((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}

		else if(f->getDirection()=="FORWARD") {

			vView *= vel;
			vPos += vView;
			setProp((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}
				
		else if(f->getDirection()=="LEFT") {
			
			vRight *= vel;
			vPos -= vRight;
			setProp((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}

		else if(f->getDirection()=="RIGHT") {
			
			vRight *= vel;
			vPos += vRight;
			setProp((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}
		else if (f->getDirection() == "UP") {
		
			vUp *= vel;
			vPos += vUp;
			setProp((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}
		else if (f->getDirection() == "DOWN") {

			vUp *= vel;
			vPos -= vUp;
			setProp((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}

		if (m_LookAt) {

			vec4 v = m_Float4Props[LOOK_AT_POINT];
			setProp((Float4Property)LOOK_AT_POINT, v.x, v.y, v.z, 1.0f);
		}

		result.set(m_Float4Props[POSITION].x, m_Float4Props[POSITION].y, m_Float4Props[POSITION].z); 
		m_Event.setData(&result);
		EVENTMANAGER->notifyEvent("CAMERA_POSITION", m_Name,"", &m_Event);
	}
}


// Physics
bool 
Camera::isDynamic() 
{
	return m_IsDynamic;
}
			
void 
Camera::setDynamic(bool value)
{
	m_IsDynamic = value;
}

void 
Camera::setPositionOffset (float value)
{
	m_PositionOffset = value;
}


// ================ OLD CODE ==========================


/*
void
CPerspCamera::move(float direction)
{
	vec3 look(m_LookAtPoint);

	look -= m_Position;
	look.normalize ();

	m_Position.x += look.x * direction;
	m_Position.y += look.y * direction;
	m_Position.z += look.z * direction;

	m_LookAtPoint.x += look.x * direction;
	m_LookAtPoint.y += look.y * direction;
	m_LookAtPoint.z += look.z * direction;

}

void
CPerspCamera::upAndDown(float direction)
{
	m_Position.y = direction;
	m_LookAtPoint.y = direction;
}

void 
CPerspCamera::strafe(float direction)
{
	vec3 look(m_LookAtPoint);

	look -= m_Position;
	look.normalize ();

	vec3 cross;
	cross = look.cross(m_UpVector);

	m_Position.x += cross.x * direction;
	m_Position.z += cross.z * direction;

	m_LookAtPoint.x += cross.x * direction;
	m_LookAtPoint.z += cross.z * direction;
}

void 
CPerspCamera::rotate(float angle, float x, float y, float z)
{
	vec3 look(m_LookAtPoint);

	float cosAngle = (float)cos(DegToRad (angle));
	float sinAngle = (float)sin(DegToRad (angle));

	look -= m_Position;
	look.normalize();

	vec3 tmp;

	tmp.x = (cosAngle + (1 - cosAngle) * x) * look.x;
	tmp.x += ((1 - cosAngle) * x * y - z * sinAngle) * look.y;
	tmp.x += ((1 - cosAngle) * x * z + y * sinAngle) * look.z;

	tmp.y = ((1 - cosAngle) * x * y + z * sinAngle) * look.x;
	tmp.y += (cosAngle + (1 - cosAngle) * y) * look.y;
	tmp.y += ((1 - cosAngle) * y * z - x * sinAngle) * look.z;

	tmp.z = ((1 - cosAngle) * x * z - y * sinAngle) * look.x;
	tmp.z += ((1 - cosAngle) * y * z + x * sinAngle) * look.y;
	tmp.z += (cosAngle + (1 - cosAngle) * z) * look.z;

	m_LookAtPoint.x = m_Position.x + tmp.x;
	m_LookAtPoint.y = m_Position.y + tmp.y;
	m_LookAtPoint.z = m_Position.z + tmp.z;
}

void 
CPerspCamera::rotate(float angle, vec3 &axis)
{
	rotate(angle, axis.x, axis.y, axis.z);
}
*/

//void 
//Camera::activate (void)
//{
//	m_Active = true;
//}
//
//void 
//Camera::deactivate (void)
//{
//	m_Active = false;
//}
//
//bool 
//Camera::isActive (void)
//{
//	return m_Active;
//}

//vec3& 
//Camera::getUpVector ()
//{
//	result = vec3(m_Float4Props[NORMALIZED_UP_VEC].x, m_Float4Props[NORMALIZED_UP_VEC].y, m_Float4Props[NORMALIZED_UP_VEC].z);
//	return result;
//}
//
//vec3& 
//Camera::getViewVector ()
//{
//	result = vec3(m_Float4Props[NORMALIZED_VIEW_VEC].x, m_Float4Props[NORMALIZED_VIEW_VEC].y, m_Float4Props[NORMALIZED_VIEW_VEC].z);
//	return result;
//}
//
//vec3& 
//Camera::getPosition ()
//{
//	result = vec3(m_Float4Props[POSITION].x, m_Float4Props[POSITION].y, m_Float4Props[POSITION].z);
//	return result;
//}
//
//
//vec3&
//Camera::getRightVector()
//{
//	result = vec3(m_Float4Props[NORMALIZED_RIGHT_VEC].x, m_Float4Props[NORMALIZED_RIGHT_VEC].y, m_Float4Props[NORMALIZED_RIGHT_VEC].z);
//	return result;
//}




//void 
//Camera::setLookAtPoint (float x, float y, float z)
//{
//	m_LookAtPoint.x = x;
//	m_LookAtPoint.y = y;
//	m_LookAtPoint.z = z;
//}

//void 
//Camera::setLookAtPoint (const vec3& v)
//{
//	setLookAtPoint (v.x, v.y, v.z);
//}

//void 
//Camera::setPosition (float x, float y, float z)
//{
//	setProp((Float4Property)POSITION, x, y, z, 1.0f);
//	//m_Position.x = x;
//	//m_Position.y = y;
//	//m_Position.z = z;
//}
//
//void 
//Camera::setPosition (const vec3& v)
//{
//	setPosition (v.x, v.y, v.z);
//}
//
//void 
//Camera::setUpVector (float x, float y, float z)
//{
//	setProp((Float4Property)UP_VEC, x, y, z, 0.0f);
//}
//
//void 
//Camera::setUpVector (const vec3& v)
//{
//	setProp((Float4Property)UP_VEC, v.x, v.y, v.z, 0.0f);
//}
//

//void 
//Camera::setViewVector (float x, float y, float z)
//{
//	m_Spherical.set(x,y,z);
//	//OR
//	//vec3 aux(x,y,z);
//	//aux.normalize();
//	//m_ElevationAngle = asin(aux.y);
//	//if (aux.z >= 0)
//	//	m_ZXAngle = asin(aux.x / sqrt(aux.x*aux.x + aux.z*aux.z));
//	//else
//	//	m_ZXAngle = (float)M_PI - asin(aux.x / sqrt(aux.x*aux.x + aux.z*aux.z));
//	// END OR
//	setVectorsFromSpherical();
//	buildViewMatrix();
//}


//void 
//Camera::setViewVector (const vec3& v)
//{
//	setViewVector (v.x, v.y, v.z);
//}

//void
//Camera::setFarPlane (float afar)
//{
//	m_Far = afar;
//	if (false == m_IsOrtho) {
//		setPerspective (m_FOV, m_Near, m_Far);
//	} else {
//		setOrtho (m_Left, m_Right, m_Bottom, m_Top, m_Near, m_Far);
//	}
//}
//
//
//void
//Camera::setNearPlane (float anear)
//{
//	m_Near = anear;
//	if (false == m_IsOrtho) {
//		setPerspective (m_FOV, m_Near, m_Far);
//	} else {
//		setOrtho (m_Left, m_Right, m_Bottom, m_Top, m_Near, m_Far);
//	}
//}
//nau::math::vec3&
//Camera::getLookAtPoint()
//{
//	return m_LookAtPoint;
//}

//void
//Camera::setLookAtPoint(float x, float y, float z) 
//{
//	vec3 aux;
//
//	m_LookAtPoint.set(x,y,z);
//	aux.set(-m_Position.x, -m_Position.y, -m_Position.z);
//	aux += m_LookAtPoint;
//	setViewVector(aux.x,aux.y,aux.z);
//}

//void 
//Camera::setLookAtPoint(const vec3 &v) 
//{
//	setLookAtPoint(v.x, v.y, v.z);
//}
//std::string&
//Camera::getName (void)
//{
//	return m_Name;
//}


//void 
//Camera::getPropTypeAndId(std::string &s, nau::Enums::DataType *dt , int *id) 
//{
//
//	Attribute a = Attribs.get(s);
//	*id = a.mId;
//
//	if (a.mId != -1) {
//
//		*dt = a.mType;
//	}
//
//	return;
	//// value returned in case of an invalid string
	//*id = -1;

	//for (int i = 0; i < COUNT_MAT4PROPERTY; i++) {

	//	if (s == Mat4String[i]) {
	//	
	//		*id = i;
	//		*dt = Enums::MAT4;
	//		return;
	//	}
	//}

	//for (int i = 0; i < COUNT_FLOATPROPERTY; i++) {

	//	if (s == FloatString[i]) {
	//	
	//		*id = i;
	//		*dt = Enums::FLOAT;
	//		return;
	//	}
	//}

	//for (int i = 0; i < COUNT_FLOAT4PROPERTY; i++) {

	//	if (s == Float4String[i]) {
	//	
	//		*id = i;
	//		*dt = Enums::VEC4;
	//		return;
	//	}
	//}
//}

//const std::string &
//Camera::getPropfString(Camera::FloatProperty prop) 
//{
//	return FloatString[prop];
//}
//
//			
//const std::string &
//Camera::getPropfvString(Camera::Float4Property prop)
//{
//	return Float4String[prop];
//}
//
//			
//const std::string & 
//Camera::getPropm4String(Camera::Mat4Property prop)
//{
//	return Mat4String[prop];
//}


//void 
//Camera::setLookAt(bool flag)
//{
//	m_LookAt = flag;
//}
//
//bool 
//Camera::getLookAt()
//{
//	return m_LookAt;
//}


//void
//Camera::setZXAngle(float angle)
//{
//	m_Spherical.setAlpha(angle);
//	setVectorsFromSpherical();
//}
//
//void
//Camera::setElevationAngle(float angle)
//{
//	m_Spherical.setBeta(angle);
//	setVectorsFromSpherical();
//}


//float
//Camera::getZXAngle()
//{
//	return m_Spherical.alpha();//m_ZXAngle;
//}
//
//
//float 
//Camera::getElevationAngle()
//{
//	return m_Spherical.beta();//m_ElevationAngle;
//}

//void
//Camera::setProjectionType(CameraType ct) {
//
//	if (ct == ORTHO)
//		m_IsOrtho = true;
//	else
//		m_IsOrtho = false;
//
//	buildProjectionMatrix();
//	buildProjectionViewMatrix();
//	buildTS05PVMMatrix();
//}

//unsigned int Camera::getProjectionType() {
//
//	if (m_IsOrtho)
//		return(ORTHO);
//	else
//		return(PERSPECTIVE);
//}