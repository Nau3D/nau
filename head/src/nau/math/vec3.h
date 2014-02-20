#ifndef VEC3_H
#define VEC3_H

//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/binary_iarchive.hpp>

namespace nau
{
	namespace math
	{

		//! \brief This class defines a vector in 3D space.
		//!
		//! \author Eduardo Marques
		//!	
		//! A vector defines both a magnitude (length) and a
		//! direction. One use for the vec3 class is as a data type for the
		//! entities in physics equations. A vector can also be use to
		//! define a point in 3D space ( a point is essentially a vector
		//! that originates at (0,0,0) ).
		//!		
		//! vec3 defines the three vector components (x,y,z) as data
		//! members and has several vector operations as member functions.
		class vec3 {
			//friend class boost::serialization::access;
		 public:
		  
		  /// Static copies of the unit vectors
		  static const vec3 UNIT_X, 
		    UNIT_Y, 
		    UNIT_Z, 
		    NEGATIVE_UNIT_X,
		    NEGATIVE_UNIT_Y,
		    NEGATIVE_UNIT_Z; 
		  
			//! x, y, z vector components 
			float x,y,z;							
		   
			/// \name Constructors
			//@{
		   
			//! Default class constructor. Initialize vector to (0,0,0,0)
			explicit vec3 (): x(0.0f), y(0.0f), z(0.0f) {};
		   
			//! Complete class constructor
			explicit vec3 (float x, float y, float z): x(x), y(y), z(z) {};
		   
			//! Copy constructor. 
			vec3 (const vec3 &v);
		   
			//! Class destructor 
			~vec3 () {};								
			
			//@}
		   
			//! \name Vector Accessing and Copying
			//@{
		   
			//! Make this vector a copy of the vector <i>v</i>
			void copy (const vec3 &v);

			//! Return a new vector with the same contents a this vector
			vec3 * clone () const;

			//! Initialize ou change the vector's components
			void set (float x,float y, float z);	

		   
			// Assignment operator
			const vec3& operator = (const vec3 &v);
		   
			//@}
		   
			//! \name Overloaded Arithmetic Operators
			//@{							
		   
			//! Vector addition
			const vec3& operator += (const vec3 &v); 		
		   
			//! Vector subtraction
			const vec3& operator -= (const vec3 &v);
		   
			//! Vector negation
			const vec3& operator - (void);				

			//! Vector scaling
			const vec3& operator *= (float t);
		   
			//! Scalar division
			const vec3& operator /= (float t);					
		   
			//@}
		   
		   
			//! \name Overloaded comparation
			//@{
			
			//! Equal
			bool operator == (const vec3 &v) const;
			
			//! Diferent
			bool operator != (const vec3 &v) const;

			//@}

			//! \name Methods
			//@{
		   
			//! Length of vector
			float length () const;
		   
			//! Length without the square root
			float sqrLength () const;					       
		   
			//! distance between this vector and vector <i>v</i>
			float distance (const vec3 &v) const;		
			
			//! Normalize this vector
			void normalize ();
		   
			//! Return the unit vector of this vector
			const vec3 unitVector () const;					
		   
			//! Dot product between this vector and vector <i>v</i>
			float dot (const vec3 &v) const;	
		   
			//! Cross product between this vector and vector <i>v</i>
			const vec3 cross (const vec3 &v) const;
		   
			//! Angle between two vectors.
			float angle (vec3 &v) const;
		   
			//! Interpolated vector between this vector and vector <i>v</i> at
			//! position alpha (0.0 <= alpha <= 1.0)
			const vec3 lerp (const vec3 &v, float alpha) const;   
		   
			//! Add another vector to this one
			void add (const vec3 &v);
		   
			//! Scalar multiplication
			void scale (float a);				
		   
			//! Vector Equality
			bool equals (const vec3 &v, float tolerance=-1.0f) const;
		   
			//@}			
			

		public:
			/*
			* boost serialization interface;
			*/	
			//template<class Archive>
			//void serialize (Archive &ar, const unsigned int version)
			//{
			//	ar & x;
			//	ar & y;
			//	ar & z;
			//}
			//END: boost serialization interface
		};
	};
};
#endif // VEC3_H
