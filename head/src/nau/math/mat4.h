#ifndef MAT4_H
#define MAT4_H

#include <nau/math/vec4.h>
#include <nau/math/vec3.h>

namespace nau
{
	namespace math
	{
		//! \brief Simple 4x4 Matrix class
		//!
		//! This class uses column-major matrix order for OpenGL compatibility. 
		class mat4 
		{
		 private:
		   
			float m_Matrix[16];
			float m_SubMat3[9];

		 public:
		 
			//! \name Constructors
			//@{
		  
			//! Default Constructor
			mat4();
		  
			//! Constructor from a float array, usually a GL Matrix
			mat4 (const float *matrix);
		  
			//! Destructor
			~mat4 ();

			//@}
		  
			//! \name Matrix access and copy
			//@{
		  
			//! Get matrix value at position (i,j).
			//! M(i,j) = line i, column j  
			float at (int i, int j) const;
		  
			//! Set the matrix value at position (i,j).
			//! M(i,j) = line i, column j
			void set (int i, int j, float value);
		  
			//! Get the matrix data as a constant float *. This is usable as a
			//! GL matrix
			const float * getMatrix() const;
			const float * getSubMat3() ;
		   
			//! Set the matrix data, usually from a GL matrix
			void setMatrix (const float *matrix);
		   
			//! Set mat4 to the identity matrix
			void setIdentity ();
		  
			//! Copy the contents of matrix <i>m</i> into this matrix
			void copy (const mat4 &m);

			//! Create a new matrix with the same contents of this matrix
			mat4 * clone ();
		  
			//@}

			//! \name Operators
			//@{
		  
			//! Add a mat4 to this one
			mat4 &operator += (const mat4 &m);
		  
			//! Subtract a mat4 from this one
			mat4 &operator -= (const mat4 &m);
		  
			//! Multiply the mat4 by another mat4
			mat4 &operator *= (const mat4 &m);
		  
			//! Multiply the mat4 by a float
			mat4 &operator *= (float f);
		  
			//@}
		  
			//! \name Matrix operations
			//@{
		  
			//! Transpose the mat4
			void transpose();
			//! Invert the mat4
			void invert();
		  
			//! Multiply the matrix by another mat4
			void multiply (const mat4 &m);
		   
			//! Transform (i.e. multiply) a vector by this matrix.
			void transform (vec4 &v) const;
			void transform (vec3 &v) const;
			void transform3 (vec4 &v) const;
		   
			//! Check this mat4 for equality with another mat4.
			//!
			//! \param tolerance The tolerance value for the floating point
			//! comparison. The default value uses the floating point precision
			//! unit.
			bool equals (const mat4 &m, float tolerance=-1.0) const;
		   
			//@}
		  
		 //private:
		  
			//! Revoked copy Constructor
			mat4 (const mat4 &m);

			//! Revoked assignment operator
			const mat4 & operator = (const mat4 &m);

		};
	};
};
#endif // MAT4_H
