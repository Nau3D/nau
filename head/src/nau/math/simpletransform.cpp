#include <nau/math/utils.h>
#include <nau/math/simpletransform.h>


#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <cmath>


using namespace nau::math;


SimpleTransform::SimpleTransform(): m_Matrix()
{
//	m_Matrix = new mat4(); 
}

SimpleTransform::~SimpleTransform ()
{
}

const mat4 & 
SimpleTransform::getMat44 () const
{
   return m_Matrix;
}

void
SimpleTransform::clone(ITransform *s) {

	//SimpleTransform *st = (SimpleTransform *)s;
	m_Matrix.copy(s->getMat44()/*st->m_Matrix*/);
}
//void
//SimpleTransform::clone(ITransform &s) {
//
//	SimpleTransform &st = (SimpleTransform)s;
//	m_Matrix.copy(st.m_Matrix);
//}

ITransform *
SimpleTransform::clone() {

	SimpleTransform *st = new SimpleTransform();
	st->m_Matrix.copy(m_Matrix);

	return(st);
}
   
void 
SimpleTransform::setMat44 (mat4 &m)
{
  this->m_Matrix.copy(m);
}
   

bool
SimpleTransform::isIdentity() {

	float p_Identity[] = {	1.0f, 0.0f, 0.0f, 0.0f,
							0.0f, 1.0f, 0.0f, 0.0f,
							0.0f, 0.0f, 1.0f, 0.0f,
							0.0f, 0.0f, 0.0f, 1.0f};

	if (memcmp(m_Matrix.getMatrix(),p_Identity, 16 * sizeof(float)))
		return false;
	else
		return true;
}


void 
SimpleTransform::setTranslation (float x, float y, float z)
{
	m_Translation.set (x, y, z);
	m_Matrix.setIdentity();
    m_Matrix.set (3, 0, x);
	m_Matrix.set (3, 1, y);
	m_Matrix.set (3, 2, z);
}


void 
SimpleTransform::setTranslation (const vec3 &v)
{
	m_Matrix.setIdentity();
	setTranslation (v.x, v.y, v.z);
}
   
void 
SimpleTransform::setRotation (float angle, float x, float y, float z)
{
	float radAngle = nau::math::DegToRad(angle);
	float co = cos(radAngle);
	float si = sin(radAngle);
	float x2 = x*x;
	float y2 = y*y;
	float z2 = z*z;

	m_Matrix.set(0,0, x2 + (y2 + z2) * co); 
	m_Matrix.set(1,0, x * y * (1 - co) - z * si);
	m_Matrix.set(2,0, x * z * (1 - co) + y * si);
	m_Matrix.set(3,0, 0.0f);

	m_Matrix.set(0,1, x * y * (1 - co) + z * si);
	m_Matrix.set(1,1, y2 + (x2 + z2) * co);
	m_Matrix.set(2,1, y * z * (1 - co) - x * si);
	m_Matrix.set(3,1, 0.0f);

	m_Matrix.set(0,2, x * z * (1 - co) - y * si);
	m_Matrix.set(1,2, y * z * (1 - co) + x * si);
	m_Matrix.set(2,2, z2 + (x2 + y2) * co);
	m_Matrix.set(3,2, 0.0f);

	m_Matrix.set(0,3, 0.0f);
	m_Matrix.set(1,3, 0.0f);
	m_Matrix.set(2,3, 0.0f);
	m_Matrix.set(3,3, 1.0f);

}
   
void 
SimpleTransform::setRotation (float angle, const vec3 &v)
{
   //TODO: Implement Method
//	m_Matrix.setIdentity();
	setRotation(angle, v.x, v.y, v.z);
}
   
void 
SimpleTransform::setScale (float amount)
{
	m_Matrix.setIdentity();
 	m_Matrix.set (0, 0, amount);
	m_Matrix.set (1, 1, amount);
	m_Matrix.set (2, 2, amount);
 
}
   
void 
SimpleTransform::setScale (vec3 &v)
{
   //TODO: Implement Method
	m_Matrix.setIdentity();
 	m_Matrix.set (0, 0, v.x);
	m_Matrix.set (1, 1, v.y);
	m_Matrix.set (2, 2, v.z);
}


void 
SimpleTransform::translate (float x, float y, float z)
{
	mat4 translation;

	translation.set (3, 0, x);
	translation.set (3, 1, y);
	translation.set (3, 2, z);

	m_Matrix *= translation;
}

void 
SimpleTransform::translate (const vec3 &v)
{
   translate (v.x, v.y, v.z);
}
   
void 
SimpleTransform::rotate (float angle, float x, float y, float z)
{
	mat4 matrix;
  	float radAngle = nau::math::DegToRad(angle);
	float co = cos(radAngle);
	float si = sin(radAngle);
	float x2 = x*x;
	float y2 = y*y;
	float z2 = z*z;

	matrix.set(0,0, x2 + (y2 + z2) * co); 
	matrix.set(1,0, x * y * (1 - co) - z * si);
	matrix.set(2,0, x * z * (1 - co) + y * si);
	matrix.set(3,0, 0.0f);

	matrix.set(0,1, x * y * (1 - co) + z * si);
	matrix.set(1,1, y2 + (x2 + z2) * co);
	matrix.set(2,1, y * z * (1 - co) - x * si);
	matrix.set(3,1, 0.0f);

	matrix.set(0,2, x * z * (1 - co) - y * si);
	matrix.set(1,2, y * z * (1 - co) + x * si);
	matrix.set(2,2, z2 + (x2 + y2) * co);
	matrix.set(3,2, 0.0f);

	matrix.set(0,3, 0.0f);
	matrix.set(1,3, 0.0f);
	matrix.set(2,3, 0.0f);
	matrix.set(3,3, 1.0f);

	m_Matrix *= matrix;
}
   
void 
SimpleTransform::rotate (float angle, vec3 &v)
{
	rotate(angle, v.x, v.y, v.z);
}
 

void 
SimpleTransform::scale (float amount)
{
	mat4 scale;

	scale.set (0, 0, amount);
	scale.set (1, 1, amount);
	scale.set (2, 2, amount);

	m_Matrix *= scale;
}


void 
SimpleTransform::scale (vec3 &v)
{
	mat4 scale;

	scale.set (0, 0, v.x);
	scale.set (1, 1, v.y);
	scale.set (2, 2, v.z);

	m_Matrix *= scale;

}
 
void
SimpleTransform::scale (float x, float y, float z)
{
	mat4 scale;

	scale.set (0, 0, x);
	scale.set (1, 1, y);
	scale.set (2, 2, z);

	m_Matrix *= scale;

}
 

void 
SimpleTransform::compose (const ITransform &t)
{
  const mat4 &ComposeMatrix = t.getMat44();
  
  mat4 &m = this->m_Matrix;
  
  m.multiply(ComposeMatrix);
}


void
SimpleTransform::transpose() {

	m_Matrix.transpose();
}


void
SimpleTransform::invert (void)
{
	m_Matrix.invert();
}


void
SimpleTransform::setIdentity (void)
{
	m_Matrix.setIdentity();
}


std::string 
SimpleTransform::getType (void) const
{
	return "SimpleTransform";
}



const vec3 & 
SimpleTransform::getTranslation ()
{
	m_Translation.set (m_Matrix.at (3, 0), m_Matrix.at (3, 1), m_Matrix.at (3, 2));
	return m_Translation;
}
   
//float 
//SimpleTransform::getRotationAngle ()
//{
//   //TODO: Implement Method
//	return 0.0f;
//}
//   
//const vec3 & 
//SimpleTransform::getRotationAxis ()
//{
//   //TODO: Implement Method
//	return *(new vec3());
//} 
   
//float 
//SimpleTransform::getScale ()
//{
//   //TODO: Implement Method
//	return 0.0f;
//}
//

