#ifndef MAT3_H
#define MAT3_H

#include <nau/math/vec3.h>

namespace nau
{
	namespace math
	{
		//! \brief Simple 3x3 Matrix class
		//!
		//! This class uses column-major matrix order for OpenGL compatibility. 
		class mat3 
		{
		 private:
		   
			float m_Matrix[9];

		 public:
		 
			//! \name Constructors
			//@{
		  
			//! Default Constructor
			mat3();
		  
			//! Constructor from a float array, usually a GL Matrix
			mat3 (const float *matrix);
		  
			//! Destructor
			~mat3 ();

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
		   
			//! Set the matrix data, usually from a GL matrix
			void setMatrix (const float *matrix);
		   
			//! Set mat3 to the identity matrix
			void setIdentity ();
		  
			//! Copy the contents of matrix <i>m</i> into this matrix
			void copy (const mat3 &m);

			//! Create a new matrix with the same contents of this matrix
			mat3 * clone ();
		  
		  
			//@}
		  
			//! \name Matrix operations
			//@{
		  
			//! Transpose the mat3
			void transpose();
			//! Invert the mat3
			void invert();
		  		   
			//@}
		  
		 //private:
		  
			//! Revoked copy Constructor
			mat3 (const mat3 &m);

			//! Revoked assignment operator
			const mat3 & operator = (const mat3 &m);

		};
	};
};
#endif // MAT3_H
