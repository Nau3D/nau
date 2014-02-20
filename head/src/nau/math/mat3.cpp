#include <nau/math/mat3.h>
#include <nau/math/utils.h>

#include <cmath>

using namespace nau::math;

// inline operator to get the index of the matrix value in position (i,j)
// i.e. line i, column j. This allow for easy translation between the math
// notation for matrices and the OpenGL reversed notation. 
static inline int 
M3(int i, int j)
{ 
   return (i*3+j); 
};

// Default constructor. Sets an identity matrix.
mat3::mat3() 
{
	setIdentity();
}

// Constructor from a float array, usually a GL Matrix
mat3::mat3 (const float *matrix)
{
   this->setMatrix (matrix);
}

mat3::~mat3()
{
}
  
// Get matrix value at position (i,j).
// M3(i,j) = line i, column j  
float 
mat3::at (int i, int j) const
{
   return (this->m_Matrix[M3(i,j)]);
}
  
// Set the matrix value at position (i,j).
// M(i,j) = line i, column j
void 
mat3::set (int i, int j, float value)
{
   this->m_Matrix[M3(i,j)] = value;
}
  
// Get the matrix data as a constant float *. This is usable as a
// GL matrix
const float * 
mat3::getMatrix() const
{
   return (this->m_Matrix);
}



// Set the matrix data, usually from a GL matrix
void 
mat3::setMatrix (const float *matrix)
{
   for (int i=0; i < 9; ++i) {m_Matrix[i] = matrix[i]; }
}
   
// Set mat3 to the identity matrix
void 
mat3::setIdentity ()
{
   for (int i=0; i<3; i++) {
      for (int j=0; j<3; j++) {	 
			m_Matrix[M3(i,j)] = (i == j) ? 1.0f : 0.0f;
      }
   }
}
  
// Copy the contents of matrix <i>m</i> into this matrix
void 
mat3::copy (const mat3 &m)
{
   this->setMatrix (m.m_Matrix);
}

// Create a new matrix with the same contents of this matrix
mat3 * 
mat3::clone ()
{
   return (new mat3 (this->m_Matrix));
   
}


// Transpose this matrix
void 
mat3::transpose()
{
	float aux;
	for (int i=0; i < 3; ++i) {
		for (int j=i+1; j < 3; ++j) {
			aux = m_Matrix[M3(i,j)];
			m_Matrix[M3(i,j)] = m_Matrix[M3(j,i)];
			m_Matrix[M3(j,i)] = aux;
		}
   }
}


// Invert this matrix
void 
mat3::invert()
{
	float m[9], det, invDet;

	det =	m_Matrix[M3(0,0)] * (m_Matrix[M3(2,2)] * m_Matrix[M3(1,1)] - m_Matrix[M3(2,1)] * m_Matrix[M3(1,2)]) -
			m_Matrix[M3(1,0)] * (m_Matrix[M3(2,2)] * m_Matrix[M3(0,1)] - m_Matrix[M3(2,1)] * m_Matrix[M3(0,2)]) +
			m_Matrix[M3(2,0)] * (m_Matrix[M3(1,2)] * m_Matrix[M3(0,1)] - m_Matrix[M3(1,1)] * m_Matrix[M3(0,2)]);
	
	invDet = 1.0f / det;

	m[M3(0,0)] =  (m_Matrix[M3(2,2)] * m_Matrix[M3(1,1)] - m_Matrix[M3(2,1)] * m_Matrix[M3(1,2)])/invDet;
	m[M3(0,1)] = -(m_Matrix[M3(2,2)] * m_Matrix[M3(0,1)] - m_Matrix[M3(2,1)] * m_Matrix[M3(0,2)])/invDet;
	m[M3(0,2)] =  (m_Matrix[M3(1,2)] * m_Matrix[M3(0,1)] - m_Matrix[M3(1,1)] * m_Matrix[M3(0,2)])/invDet;

	m[M3(1,0)] = -(m_Matrix[M3(2,2)] * m_Matrix[M3(1,0)] - m_Matrix[M3(2,0)] * m_Matrix[M3(1,2)])/invDet;
	m[M3(1,1)] =  (m_Matrix[M3(2,2)] * m_Matrix[M3(0,0)] - m_Matrix[M3(2,0)] * m_Matrix[M3(0,2)])/invDet;
	m[M3(1,2)] = -(m_Matrix[M3(1,2)] * m_Matrix[M3(0,0)] - m_Matrix[M3(1,0)] * m_Matrix[M3(0,2)])/invDet;

	m[M3(2,0)] =  (m_Matrix[M3(2,1)] * m_Matrix[M3(1,0)] - m_Matrix[M3(2,0)] * m_Matrix[M3(1,1)])/invDet;
	m[M3(2,1)] = -(m_Matrix[M3(2,1)] * m_Matrix[M3(0,0)] - m_Matrix[M3(2,0)] * m_Matrix[M3(0,1)])/invDet;
	m[M3(2,2)] =  (m_Matrix[M3(1,1)] * m_Matrix[M3(0,0)] - m_Matrix[M3(1,0)] * m_Matrix[M3(0,1)])/invDet;

	this->setMatrix(m);
}
  
// Private copy Constructor
mat3::mat3 (const mat3 &m)
{
   this->setMatrix (m.m_Matrix);
}

// Private assignment operator
const mat3 & 
mat3::operator = (const mat3 &m)
{
   this->setMatrix (m.m_Matrix);
   
   return (*this);
}
