#include "nau/scene/camera.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/event/eventFactory.h" 
#include "nau/event/cameraMotion.h"
#include "nau/event/cameraOrientation.h"
#include "nau/geometry/boundingbox.h"
#include "nau/geometry/mesh.h"
#include "nau/material/materialgroup.h"
#include "nau/math/matrix.h"
#include "nau/math/utils.h"
#include "nau/render/irenderer.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

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
	//Attribs.add(Attribute(NORMALIZED_VIEW_VEC, "NORMALIZED_VIEW_VEC", Enums::DataType::VEC4, true,new vec4(0.0f, 0.0f, -1.0f, 0.0f)));
	Attribs.add(Attribute(UP_VEC, "UP", Enums::DataType::VEC4, false, new vec4(0.0f, 1.0f, 0.0f, 0.0f)));
	Attribs.add(Attribute(NORMALIZED_UP_VEC, "NORMALIZED_UP" ,Enums::DataType::VEC4, true, new vec4(0.0f, 1.0f, 0.0f, 0.0f)));
	Attribs.add(Attribute(NORMALIZED_RIGHT_VEC, "NORMALIZED_RIGHT" ,Enums::DataType::VEC4, true, new vec4(1.0f, 0.0f, 0.0f, 0.0f)));
	Attribs.add(Attribute(LOOK_AT_POINT, "LOOK_AT_POINT" ,Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, -1.0f, 1.0f)));
	// MAT4
	Attribs.add(Attribute(VIEW_MATRIX, "VIEW_MATRIX",Enums::DataType::MAT4, true));
	Attribs.add(Attribute(PROJECTION_MATRIX, "PROJECTION_MATRIX",Enums::DataType::MAT4, true));
	Attribs.add(Attribute(VIEW_INVERSE_MATRIX, "VIEW_INVERSE_MATRIX",Enums::DataType::MAT4, true));
	Attribs.add(Attribute(PROJECTION_INVERSE_MATRIX, "PROJECTION_INVERSE_MATRIX", Enums::DataType::MAT4, true));

	Attribs.add(Attribute(PROJECTION_VIEW_MATRIX, "PROJECTION_VIEW_MATRIX",Enums::DataType::MAT4, true));
	Attribs.add(Attribute(TS05_PVM_MATRIX, "TS05_PVM_MATRIX",Enums::DataType::MAT4, true));
	// FLOAT
	Attribs.add(Attribute(FOV, "FOV",Enums::DataType::FLOAT, false, new float(60.0f)));
	Attribs.add(Attribute(NEARP, "NEAR",Enums::DataType::FLOAT, false, new float(1.0f)));
	Attribs.add(Attribute(FARP, "FAR",Enums::DataType::FLOAT, false, new float(10000.0f)));
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

	NAU->registerAttributes("CAMERA", &Attribs);

	return true;
}


AttribSet Camera::Attribs;
bool Camera::Inited = Init();

//void
//Camera::setDefault()
//{
//	Attribs.initAttribInstanceFloatArray(m_FloatProps);
//	Attribs.initAttribInstanceVec4Array(m_Float4Props);
//	Attribs.initAttribInstanceEnumArray(m_EnumProps);
//}


Camera::Camera (const std::string &name) :
	SceneObject(),

	m_IsDynamic (false),
	m_pViewport (0),
	m_LookAt(false),
	//m_LookAtPoint(0.0f, 0.0f, 0.0f),
	m_PositionOffset (0.0f)
	//m_IsOrtho (false)
{
	//setDefault();
	registerAndInitArrays(Attribs);
	m_Id = 0;
	m_Name = name;
	m_pViewport = NAU->getDefaultViewport();

	buildViewMatrix();
	buildInverses();

	m_StaticCondition = false;

	m_BoundingVolume = new BoundingBox;
	setVectorsFromSpherical();

	// Adding a Mesh with the frustum lines
	Mesh *renderable =  (Mesh *)RESOURCEMANAGER->createRenderable("Mesh", m_Name, "Camera");
	//int drawPrimitive = IRenderer::Attribs.getID("LINES");
	//renderable->setDrawingPrimitive(drawPrimitive/*nau::render::IRenderer::LINES*/);
	renderable->setDrawingPrimitive(nau::render::IRenderable::LINES);
	std::vector<VertexData::Attr> *vertices = new std::vector<VertexData::Attr>(8);
	VertexData &vertexData = renderable->getVertexData();
	vertexData.setDataFor (VertexData::getAttribIndex("position"), vertices);

	MaterialGroup *aMaterialGroup = MaterialGroup::Create(renderable, "__Emission Green");
	
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
	//aMaterialGroup->setParent (renderable);
	//aMaterialGroup->setMaterialName("__Black");
	//aMaterialGroup->setMaterialName("__Emission White");

	renderable->addMaterialGroup (aMaterialGroup);
	m_Transform = m_Mat4Props[VIEW_INVERSE_MATRIX];
	setRenderable (renderable);

	aMaterialGroup = MaterialGroup::Create(renderable, "__Emission Red");
	indices = new std::vector<unsigned int>(8);
	indices->at (0) = Camera::TOP_LEFT_NEAR;		indices->at (1) = Camera::TOP_RIGHT_NEAR;
	indices->at (2) = Camera::TOP_RIGHT_NEAR;		indices->at (3) = Camera::BOTTOM_RIGHT_NEAR;
	indices->at (4) = Camera::BOTTOM_RIGHT_NEAR;	indices->at (5) = Camera::BOTTOM_LEFT_NEAR;
	indices->at (6) = Camera::BOTTOM_LEFT_NEAR;		indices->at (7) = Camera::TOP_LEFT_NEAR;

	aMaterialGroup->setIndexList (indices);
	//aMaterialGroup->setParent (renderable);
	//aMaterialGroup->setMaterialName("__Black");
//	aMaterialGroup->setMaterialName("__Emission Red");

	renderable->addMaterialGroup (aMaterialGroup);
	setRenderable (renderable);

	IScene *s = RENDERMANAGER->createScene(name, "SceneAux");
	s->add(this);

	EVENTMANAGER->addListener("VIEWPORT_CHANGED", this);
}


Camera::~Camera (void)
{
}


void
Camera::setOrtho(float left, float right, float bottom, float top, float nearp, float farp)
{
	m_EnumProps[PROJECTION_TYPE] = ORTHO;
	m_FloatProps[LEFT] = left;
	m_FloatProps[RIGHT] = right;
	m_FloatProps[BOTTOM] = bottom;
	m_FloatProps[TOP] = top;
	m_FloatProps[NEARP] = nearp;
	m_FloatProps[FARP] = farp;

	buildProjectionMatrix();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


void 
Camera::setPerspective (float fov, float nearp, float farp)
{
	m_EnumProps[PROJECTION_TYPE] = PERSPECTIVE;
	m_FloatProps[FOV] = fov;
	m_FloatProps[NEARP] = nearp;
	m_FloatProps[FARP] = farp;

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


//void Camera::setPropm4(Mat4Property prop, mat4 &mat) 
//{
//	m_Mat4Props[prop] = mat;
//}


void 
Camera::setPropf(FloatProperty prop, float f) 
{
	
	vec3 v;

	switch(prop) {

		case ZX_ANGLE:
		case ELEVATION_ANGLE:
			m_FloatProps[prop] = f;
			setVectorsFromSpherical();
			break;
		default:
			AttributeValues::setPropf(prop, f);
			buildViewMatrix();
			buildProjectionMatrix();
			buildProjectionViewMatrix();
			buildTS05PVMMatrix();
			buildInverses();
	}
}


void 
Camera::setPropf4(Float4Property prop, vec4& aVec) {

	setPropf4(prop, aVec.x, aVec.y, aVec.z, aVec.w);
}


void
Camera::setPropf4(Float4Property prop, float x, float y, float z, float w)
{
	vec4 v;
	vec2 v2;

	v.set(x,y,z,w);

	switch(prop) {

		case POSITION:
			v.w = 1;	
			m_Float4Props[POSITION].set(v);
			m_Float4Props[LOOK_AT_POINT] = m_Float4Props[POSITION];
			m_Float4Props[LOOK_AT_POINT] += m_Float4Props[VIEW_VEC];
			break;

		case VIEW_VEC:
			v.w = 0.0f;
			v2 = Spherical::toSpherical(x,y,z);
			m_FloatProps[ZX_ANGLE] = v2.x;
			m_FloatProps[ELEVATION_ANGLE] = v2.y;
			m_Float4Props[VIEW_VEC].set(v);
			m_Float4Props[VIEW_VEC].normalize();
//			m_Float4Props[NORMALIZED_VIEW_VEC] = m_Float4Props[VIEW_VEC];

			m_Float4Props[NORMALIZED_RIGHT_VEC] = m_Float4Props[VIEW_VEC].cross(m_Float4Props[UP_VEC]);
			m_Float4Props[NORMALIZED_RIGHT_VEC].normalize();
			m_Float4Props[NORMALIZED_UP_VEC] = m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[VIEW_VEC]);
			m_Float4Props[LOOK_AT_POINT] = m_Float4Props[POSITION];
			m_Float4Props[LOOK_AT_POINT] += m_Float4Props[VIEW_VEC];
			break;

		case UP_VEC:
			v.w = 0.0f;
			m_Float4Props[UP_VEC] = v;
			m_Float4Props[UP_VEC].normalize();
			m_Float4Props[NORMALIZED_RIGHT_VEC].set(m_Float4Props[VIEW_VEC].cross(m_Float4Props[UP_VEC]));
			m_Float4Props[NORMALIZED_RIGHT_VEC].normalize();
			m_Float4Props[NORMALIZED_UP_VEC] = m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[VIEW_VEC]);
			break;

		case LOOK_AT_POINT:
			v.w = 1.0f;
			m_Float4Props[LOOK_AT_POINT].set(v);
			m_Float4Props[VIEW_VEC] = m_Float4Props[LOOK_AT_POINT];
			m_Float4Props[VIEW_VEC] -= m_Float4Props[POSITION];
			m_Float4Props[VIEW_VEC].w = 0.0f;
			m_Float4Props[VIEW_VEC].normalize();
//			m_Float4Props[NORMALIZED_VIEW_VEC].set(m_Float4Props[VIEW_VEC]);
			v2 = Spherical::toSpherical(m_Float4Props[VIEW_VEC].x,m_Float4Props[VIEW_VEC].y,m_Float4Props[VIEW_VEC].z);
			m_FloatProps[ZX_ANGLE] = v2.x;
			m_FloatProps[ELEVATION_ANGLE] = v2.y;

			m_Float4Props[NORMALIZED_RIGHT_VEC] = m_Float4Props[VIEW_VEC].cross(m_Float4Props[UP_VEC]);
			m_Float4Props[NORMALIZED_RIGHT_VEC].normalize();
			m_Float4Props[NORMALIZED_UP_VEC] = m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[VIEW_VEC]);
			break;
		default:
			AttributeValues::setPropf4(prop, x, y, z, w);
	}
	buildViewMatrix();
	buildInverses();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


void 
Camera::setPrope(EnumProperty prop, int value) 
{
	AttributeValues::setPrope(prop, value);

	buildProjectionMatrix();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();
}


void *
Camera::getProp(int prop, Enums::DataType type) {

	switch (type) {

// ARF: Check who calls this
		case Enums::MAT4:
			assert(m_Mat4Props.count(prop) > 0);
			return((void *)m_Mat4Props[prop].getMatrix());
		default:
			return AttributeValues::getProp(prop, type);
		}
}


//const mat4&
//Camera::getPropm4(Mat4Property prop)
//{
//	return m_Mat4Props[prop];
//}


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

	//std::vector<VertexData::Attr> *normals = new std::vector<VertexData::Attr>(8);
	//for (int i = 0; i < 8 ; ++i) 
	//	normals->at(i).set(0.0f, 0.0f, 0.0f);
	//vertexData.setDataFor (VertexData::getAttribIndex("normal"), normals);

	buildInverses();
	m_ResultTransform.copy(m_GlobalTransform);
	m_ResultTransform *= m_Mat4Props[VIEW_INVERSE_MATRIX];
	return (*m_Renderable);
}


IBoundingVolume*
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
	view.normalize();
	m_Float4Props[VIEW_VEC].set(view.x, view.y, view.z, 0.0f);
//	m_Float4Props[NORMALIZED_VIEW_VEC].set(view.x, view.y, view.z, 0.0f);
//	m_Float4Props[NORMALIZED_VIEW_VEC].normalize();

	vec2 vs = Spherical::toSpherical(view.x, view.y, view.z);

	up.normalize();
	m_Float4Props[UP_VEC].set(up.x, up.y, up.z, 0.0f);

	m_Float4Props[NORMALIZED_RIGHT_VEC].set(m_Float4Props[VIEW_VEC].cross(m_Float4Props[UP_VEC]));
	m_Float4Props[NORMALIZED_RIGHT_VEC].normalize();

	m_Float4Props[NORMALIZED_UP_VEC].set(m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[VIEW_VEC]));
	m_Float4Props[NORMALIZED_UP_VEC].normalize();
	//m_Float4Props[UP_VEC].set(m_Float4Props[NORMALIZED_UP_VEC]);

	vec4 v4 = m_Float4Props[POSITION];
	v4 += m_Float4Props[VIEW_VEC];
	m_Float4Props[LOOK_AT_POINT].set(v4.x, v4.y, v4.z, 1.0f);

	buildViewMatrix();
	buildInverses();
	buildProjectionViewMatrix();
	buildTS05PVMMatrix();

}


void 
Camera::buildInverses(void) {

	// This is a simpler inverse because the view matrix has a specific format
	//mat4& tmp = const_cast<mat4&>(m_ViewMatrix->getMat44());
	//mat4& tmp2 = const_cast<mat4&>(m_ViewMatrixInverse->getMat44());
	mat4 &tmp = m_Mat4Props[VIEW_MATRIX];
	mat4 &tmp2 = m_Mat4Props[VIEW_INVERSE_MATRIX];

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

	m_Mat4Props[PROJECTION_INVERSE_MATRIX] = m_Mat4Props[PROJECTION_MATRIX];
	m_Mat4Props[PROJECTION_INVERSE_MATRIX].invert();
}


void 
Camera::setVectorsFromSpherical() 
{
	// maybe only do this if there is no look at point?
	m_FloatProps[ELEVATION_ANGLE] = Spherical::capBeta(m_FloatProps[ELEVATION_ANGLE]);

	// View vector = -Z from camera
	vec3 v;
	v = Spherical::toCartesian(m_FloatProps[ZX_ANGLE], m_FloatProps[ELEVATION_ANGLE]);
	//m_Float4Props[NORMALIZED_VIEW_VEC].set( v.x, v.y, v.z, 0.0f);
	setPropf4(VIEW_VEC, v.x, v.y, v.z, 0.0f);
	//m_Float4Props[VIEW_VEC].set(v.x, v.y, v.z, 0.0f);

	//m_Float4Props[NORMALIZED_RIGHT_VEC] = m_Float4Props[VIEW_VEC].cross(m_Float4Props[UP_VEC]);
	//m_Float4Props[NORMALIZED_RIGHT_VEC].normalize();

	//m_Float4Props[NORMALIZED_UP_VEC] = m_Float4Props[NORMALIZED_RIGHT_VEC].cross(m_Float4Props[VIEW_VEC]);

	//vec4 v4 = m_Float4Props[POSITION];
	//v4 += m_Float4Props[VIEW_VEC];
	//m_Float4Props[LOOK_AT_POINT].set(v4.x, v4.y, v4.z, 1.0f);

	//buildViewMatrix();
	//buildInverses();
	//buildProjectionViewMatrix();
	//buildTS05PVMMatrix();
}


void 
Camera::buildProjectionMatrix() {

	float aspect = m_pViewport->getPropf(Viewport::RATIO);
	float f = 1.0f / tan (DegToRad(m_FloatProps[FOV] * 0.5f));

	m_Mat4Props[PROJECTION_MATRIX].setIdentity();

	mat4 &projection = m_Mat4Props[PROJECTION_MATRIX]; 

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

	if (aViewport != NULL) {

		buildProjectionMatrix();
		buildProjectionViewMatrix();
		buildTS05PVMMatrix();
		buildInverses();

	}
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

	m_Mat4Props[TS05_PVM_MATRIX] *= m_Mat4Props[PROJECTION_VIEW_MATRIX];
}

void
Camera::buildProjectionViewMatrix (void)
{
	m_Mat4Props[PROJECTION_VIEW_MATRIX].setIdentity();
	m_Mat4Props[PROJECTION_VIEW_MATRIX] *= m_Mat4Props[PROJECTION_MATRIX];
	m_Mat4Props[PROJECTION_VIEW_MATRIX] *= m_Mat4Props[VIEW_MATRIX];
}

void
Camera::buildViewMatrix (void)
{
	vec4 s,u,v;
	u = m_Float4Props[NORMALIZED_UP_VEC];
	s = m_Float4Props[NORMALIZED_RIGHT_VEC];
	v = m_Float4Props[VIEW_VEC];

	mat4 &viewMatrix = m_Mat4Props[VIEW_MATRIX]; 

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

	vec4 p = m_Float4Props[POSITION];
	m_Mat4Props[VIEW_MATRIX].translate(-p.x, -p.y, -p.z);

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

	vec4 view = aCamera->getPropf4(VIEW_VEC);
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
	mat4 &viewMatrix = m_Mat4Props[VIEW_MATRIX];

	for (int i = 0; i < 8; i++){
		viewMatrix.transform (points[i]);
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
	if (eventType == "VIEWPORT_CHANGED" && m_pViewport != NULL && m_pViewport->getName() == sender)
		updateProjection();

	//if (eventType == "DYNAMIC_CAMERA") {

	//	vec3 p = m_Transform->getTranslation();
	//	setPropf4(POSITION,p.x,p.y,p.z,1.0f);
	//	buildViewMatrix();

	//	result.set(p.x,p.y,p.z); 
	//	m_Event.setData(&result);
	//	EVENTMANAGER->notifyEvent("CAMERA_POSITION", m_Name,"", &m_Event);

	//}
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
		vec4 vView = m_Float4Props[VIEW_VEC];
		vec4 vRight = m_Float4Props[NORMALIZED_RIGHT_VEC];
		vec4 vUp = m_Float4Props[UP_VEC];

		if(f->getDirection()=="BACKWARD") {

			vView *= vel;
			vPos -=  vView;
			setPropf4((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}

		else if(f->getDirection()=="FORWARD") {

			vView *= vel;
			vPos += vView;
			setPropf4((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}
				
		else if(f->getDirection()=="LEFT") {
			
			vRight *= vel;
			vPos -= vRight;
			setPropf4((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}

		else if(f->getDirection()=="RIGHT") {
			
			vRight *= vel;
			vPos += vRight;
			setPropf4((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}
		else if (f->getDirection() == "UP") {
		
			vUp *= vel;
			vPos += vUp;
			setPropf4((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}
		else if (f->getDirection() == "DOWN") {

			vUp *= vel;
			vPos -= vUp;
			setPropf4((Float4Property)POSITION, vPos.x, vPos.y, vPos.z, 1.0f);
		}

		if (m_LookAt) {

			vec4 v = m_Float4Props[LOOK_AT_POINT];
			setPropf4((Float4Property)LOOK_AT_POINT, v.x, v.y, v.z, 1.0f);
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


