#include <nau/math/mat4.h>
#include <nau/math/utils.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

using namespace nau::math;

// inline operator to get the index of the matrix value in position (i,j)
// i.e. line i, column j. This allow for easy translation between the math
// notation for matrices and the OpenGL reversed notation. 
static inline int 
M(int i, int j)
{ 
   return (i*4+j); 
};

// Default constructor. Sets an identity matrix.
mat4::mat4() 
{
	setIdentity();
}

// Constructor from a float array, usually a GL Matrix
mat4::mat4 (const float *matrix)
{
   this->setMatrix (matrix);
}

mat4::~mat4()
{
}
  
// Get matrix value at position (i,j).
// M(i,j) = line i, column j  
float 
mat4::at (int i, int j) const
{
   return (this->m_Matrix[M(i,j)]);
}
  
// Set the matrix value at position (i,j).
// M(i,j) = line i, column j
void 
mat4::set (int i, int j, float value)
{
   this->m_Matrix[M(i,j)] = value;
}
  
// Get the matrix data as a constant float *. This is usable as a
// GL matrix
const float * 
mat4::getMatrix() const
{
   return (this->m_Matrix);
}

const float *
mat4::getSubMat3() 
{
	m_SubMat3[0] = m_Matrix[0];
	m_SubMat3[1] = m_Matrix[1];
	m_SubMat3[2] = m_Matrix[2];

	m_SubMat3[3] = m_Matrix[4];
	m_SubMat3[4] = m_Matrix[5];
	m_SubMat3[5] = m_Matrix[6];

	m_SubMat3[6] = m_Matrix[8];
	m_SubMat3[7] = m_Matrix[9];
	m_SubMat3[8] = m_Matrix[10];

	return (this->m_SubMat3);
}

// Set the matrix data, usually from a GL matrix
void 
mat4::setMatrix (const float *matrix)
{
   for (int i=0; i<16; ++i) {m_Matrix[i] = matrix[i]; }
}
   
// Set mat4 to the identity matrix
void 
mat4::setIdentity ()
{
   for (int i=0; i<4; ++i) {
      for (int j=0; j<4; ++j) {	 
			m_Matrix[M(i,j)] = (i == j) ? 1.0f : 0.0f;
      }
   }
}
  
// Copy the contents of matrix <i>m</i> into this matrix
void 
mat4::copy (const mat4 &m)
{
   this->setMatrix (m.m_Matrix);
}

// Create a new matrix with the same contents of this matrix
mat4 * 
mat4::clone ()
{
   return (new mat4 (this->m_Matrix));
   
}

// Add a mat4 to this one
mat4 &
mat4::operator += (const mat4 &m)
{
   for (int i=0; i<16; ++i) {
      this->m_Matrix[i] += m.m_Matrix[i];
   }
   
   return (*this);
}

// Subtract a mat4 from this one
mat4 &
mat4::operator -= (const mat4 &m)
{
   for (int i=0; i<16; ++i) {
      this->m_Matrix[i] += m.m_Matrix[i];
   }

   return (*this);
}
  
// Multiply the mat4 by another mat4
mat4 &
mat4::operator *= (const mat4 &m)
{
   this->multiply (m);
   
   return (*this);
}
  
// Multiply the mat4 by a float
mat4 &
mat4::operator *= (float f)
{
   for (int i=0; i<16; ++i) {
      this->m_Matrix[i] *= f;
   }
   
   return (*this);
}

// Transpose this matrix
void 
mat4::transpose()
{
   for (int i=0; i < 4; ++i) {
		for (int j=i+1; j < 4; ++j) {
			float aux = m_Matrix[M(i,j)];
			m_Matrix[M(i,j)] = m_Matrix[M(j,i)];
			m_Matrix[M(j,i)] = aux;
		}
   }
}


/************************************************************
*
*  input:
*    mat - pointer to array of 16 floats (source matrix)
*  output:
*    dst - pointer to array of 16 floats (invert matrix)
*
*************************************************************/
void Invert2(float *mat, float *dst)
{
    float    tmp[12]; /* temp array for pairs                      */
    float    src[16]; /* array of transpose source matrix */
    float    det;     /* determinant                                  */
    /* transpose matrix */
    for (int i = 0; i < 4; ++i) {
        src[i]        = mat[i*4];
        src[i + 4]    = mat[i*4 + 1];
        src[i + 8]    = mat[i*4 + 2];
        src[i + 12]   = mat[i*4 + 3];
    }
	 /* calculate pairs for first 8 elements (cofactors) */
    tmp[0]  = src[10] * src[15];
    tmp[1]  = src[11] * src[14];
    tmp[2]  = src[9]  * src[15];
    tmp[3]  = src[11] * src[13];
    tmp[4]  = src[9]  * src[14];
    tmp[5]  = src[10] * src[13];
    tmp[6]  = src[8]  * src[15];
    tmp[7]  = src[11] * src[12];
    tmp[8]  = src[8]  * src[14];
    tmp[9]  = src[10] * src[12];
    tmp[10] = src[8]  * src[13];
    tmp[11] = src[9]  * src[12];
	
    /* calculate first 8 elements (cofactors) */
    dst[0]  = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
    dst[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
    dst[1]  = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
    dst[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
    dst[2]  = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
    dst[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
    dst[3]  = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
    dst[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
    dst[4]  = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
    dst[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
    dst[5]  = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
    dst[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
    dst[6]  = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
    dst[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
    dst[7]  = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
    dst[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];
	
	 /* calculate pairs for second 8 elements (cofactors) */
    tmp[0]  = src[2]*src[7];
    tmp[1]  = src[3]*src[6];
    tmp[2]  = src[1]*src[7];
    tmp[3]  = src[3]*src[5];
    tmp[4]  = src[1]*src[6];
    tmp[5]  = src[2]*src[5];
	tmp[6]  = src[0]*src[7];
    tmp[7]  = src[3]*src[4];
    tmp[8]  = src[0]*src[6];
    tmp[9]  = src[2]*src[4];
    tmp[10] = src[0]*src[5];
    tmp[11] = src[1]*src[4];
	
	 /* calculate second 8 elements (cofactors) */
    dst[8]  = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
    dst[8] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
    dst[9]  = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
    dst[9] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
    dst[10] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
    dst[10]-= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
    dst[11] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
    dst[11]-= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
    dst[12] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
    dst[12]-= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
    dst[13] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
    dst[13]-= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
    dst[14] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
    dst[14]-= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
    dst[15] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
    dst[15]-= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];
	
    /* calculate determinant */
    det=src[0]*dst[0]+src[1]*dst[1]+src[2]*dst[2]+src[3]*dst[3];
	
    /* calculate matrix inverse */
    det = 1/det;
    for (int j = 0; j < 16; j++)
        dst[j] *= det;
}


// Invert this matrix
void 
mat4::invert()
{
   mat4 a(*this);
   mat4 b;

   Invert2(a.m_Matrix, b.m_Matrix);

 //  unsigned int r, c;
 //  unsigned int cc;
 //  unsigned int rowMax; // Points to max abs value row in this column
 //  unsigned int row;
 //  float tmp;

 //  // Go through columns
 //  for (c=0; c<4; c++) {
	//	// Find the row with max value in this column
	//	rowMax = c;
	//	for (r=c+1; r<4; r++) {
	//		if (fabs (a.at (c, r)) > fabs (a.at (c, rowMax))) {
	//			rowMax = r;
	//		}
	//	}

 //  // If the max value here is 0, we can't invert.  Set identity.
	//	if (0.0F == a.at (rowMax, c)) {
	//		setIdentity();
	//		return;
	//	}

	//	// Swap row "rowMax" with row "c"
	//	for (cc=0; cc<4; cc++) {
	//	  tmp = a.at (cc, c);
	//	  a.set (cc, c, a.at (cc, rowMax));
	//	  a.set (cc, rowMax, tmp);
	//	  tmp = b.at (cc, c);
	//	  b.set (cc, c, b.at (cc, rowMax));
	//	  b.set (cc, rowMax, tmp);
	//	}

	//	// Now everything we do is on row "c".
	//	// Set the max cell to 1 by dividing the entire row by that value
	//	tmp = a.at (c, c);
	//	for (cc=0; cc<4; cc++) {
	//	  a.set (cc, c, a.at (cc, c) / tmp);
	//	  b.set (cc, c, b.at (cc, c) / tmp);
	//	}

	//	// Now do the other rows, so that this column only has a 1 and 0's
	//	for (row = 0; row < 4; row++) {
	//		if (row != c) {
	//			tmp = a.at (c, row);
	//			for (cc=0; cc<4; cc++) {
	//			 a.set (cc, row, a.at (cc, row) - a.at (cc, c) * tmp);
	//			 b.set (cc, row, b.at (cc, row) - b.at (cc, c) * tmp);
	//			}
	//		}
	//	}
	//}

   *this = b;
}
  
// Multiply the matrix by another mat4
void 
mat4::multiply (const mat4 &m)
{
   // auxiliary matrix to keep the result
   float aux[16];
   
   // operand matrices
   const float *m1 = this->m_Matrix;
   const float *m2 = m.m_Matrix;
   
   // algebraic result: M1 * M2 (post-multiply)
   // note: loop unrolled for speed 
   aux[0] = (m1[0] * m2[0]) + (m1[4] * m2[1]) + (m1[8] * m2[2]) + \
      (m1[12] * m2[3]);
   
   aux[1] = (m1[1] * m2[0]) + (m1[5] * m2[1]) + (m1[9] * m2[2]) + \
      (m1[13] * m2[3]);

   aux[2] = (m1[2] * m2[0]) + (m1[6] * m2[1]) + (m1[10] * m2[2]) + \
      (m1[14] * m2[3]);
   
   aux[3] = (m1[3] * m2[0]) + (m1[7] * m2[1]) + (m1[11] * m2[2]) + \
      (m1[15] * m2[3]);
   
   aux[4] = (m1[0] * m2[4]) + (m1[4] * m2[5]) + (m1[8] * m2[6]) + \
      (m1[12] * m2[7]);
   
   aux[5] = (m1[1] * m2[4]) + (m1[5] * m2[5]) + (m1[9] * m2[6]) + \
      (m1[13] * m2[7]);
   
   aux[6] = (m1[2] * m2[4]) + (m1[6] * m2[5]) + (m1[10] * m2[6]) + \
      (m1[14] * m2[7]);
   
   aux[7] = (m1[3] * m2[4]) + (m1[7] * m2[5]) + (m1[11] * m2[6]) + \
      (m1[15] * m2[7]);
   
   aux[8] = (m1[0] * m2[8]) + (m1[4] * m2[9]) + (m1[8] * m2[10]) + \
      (m1[12] * m2[11]);
   
   aux[9] = (m1[1] * m2[8]) + (m1[5] * m2[9]) + (m1[9] * m2[10]) + \
      (m1[13] * m2[11]);
   
   aux[10] = (m1[2] * m2[8]) + (m1[6] * m2[9]) + (m1[10] * m2[10]) + \
      (m1[14] * m2[11]);
   
   aux[11] = (m1[3] * m2[8]) + (m1[7] * m2[9]) + (m1[11] * m2[10]) + \
      (m1[15] * m2[11]);
   
   aux[12] = (m1[0] * m2[12]) + (m1[4] * m2[13]) + (m1[8] * m2[14]) + \
      (m1[12] * m2[15]);
   
   aux[13] = (m1[1] * m2[12]) + (m1[5] * m2[13]) + (m1[9] * m2[14]) + \
      (m1[13] * m2[15]);
   
   aux[14] = (m1[2] * m2[12]) + (m1[6] * m2[13]) + (m1[10] * m2[14]) + \
      (m1[14] * m2[15]);
   
   aux[15] = (m1[3] * m2[12]) + (m1[7] * m2[13]) + (m1[11] * m2[14]) + \
      (m1[15] * m2[15]);
   
   // copy result to matrix data
   this->setMatrix(aux);
}

// Transform (i.e. multiply) a vector by this matrix.
void 
mat4::transform (vec4 &v) const
{
  vec4 aux;
  const float *m = this->m_Matrix;

  aux.x = (v.x * m[0]) + (v.y * m[4]) + (v.z * m[8]) + (v.w * m[12]);
  aux.y = (v.x * m[1]) + (v.y * m[5]) + (v.z * m[9]) + (v.w * m[13]);
  aux.z = (v.x * m[2]) + (v.y * m[6]) + (v.z * m[10]) +(v.w * m[14]);
  aux.w = (v.x * m[3]) + (v.y * m[7]) + (v.z * m[11]) +(v.w * m[15]);
  
  aux *= (1/aux.w);

  v.copy(aux);
  
  return;
}


// Transform (i.e. multiply) a vector by this matrix.
void 
mat4::transform3 (vec4 &v) const
{
  vec4 aux;
  const float *m = this->m_Matrix;

  aux.x = (v.x * m[0]) + (v.y * m[4]) + (v.z * m[8]) ;
  aux.y = (v.x * m[1]) + (v.y * m[5]) + (v.z * m[9]) ;
  aux.z = (v.x * m[2]) + (v.y * m[6]) + (v.z * m[10]);
  aux.w = v.w;
  v.copy(aux);
  
  return;
}

void 
mat4::transform (vec3 &v) const
{
  vec3 aux;
  const float *m = this->m_Matrix;

  aux.x   = (v.x * m[0]) + (v.y * m[4]) + (v.z * m[8]) + (m[12]);
  aux.y   = (v.x * m[1]) + (v.y * m[5]) + (v.z * m[9]) + (m[13]);
  aux.z   = (v.x * m[2]) + (v.y * m[6]) + (v.z * m[10]) +(m[14]);
  float w = (v.x * m[3]) + (v.y * m[7]) + (v.z * m[11]) +(m[15]);
  
  aux *= (1/w);

  v.copy(aux);
  
  return;
}


// Check this mat4 for equality with another mat4
bool 
mat4::equals (const mat4 &m, float tolerance) const
{
   for (int i=0; i<16; i++) {
      if (!FloatEqual(this->m_Matrix[i], m.m_Matrix[i], tolerance)) {
	 return false;
      }
   }
   
   return true;
}
  
// Private copy Constructor
mat4::mat4 (const mat4 &m)
{
   this->setMatrix (m.m_Matrix);
}

// Private assignment operator
const mat4 & 
mat4::operator = (const mat4 &m)
{
   this->setMatrix (m.m_Matrix);
   
   return (*this);
}
