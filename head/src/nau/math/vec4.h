#ifndef vec4_H
#define vec4_H

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
		//! direction. One use for the vec4 class is as a data type for the
		//! entities in physics equations. A vector can also be use to
		//! define a point in 3D space ( a point is essentially a vector
		//! that originates at (0,0,0) ).
		//!		
		//! vec4 defines the three vector components (x,y,z) as data
		//! members and has several vector operations as member functions.
		class vec4 {
			//friend class boost::serialization::access;
		 public:
		  
			//! x, y, z vector components 
			float x,y,z,w;							
		   
			/// \name Constructors
			//@{
		   
			//! Default class constructor. Initialize vector to (0,0,0,0)
			explicit vec4 (): x(0.0f), y(0.0f), z(0.0f), w(0.0f) {};
		   
			//! Complete class constructor
			explicit vec4 (float x, float y, float z, float w): x(x), y(y), z(z), w(w) {};
		   
			//! Copy constructor. 
			vec4 (const vec4 &v);
		   
			//! Class destructor 
			~vec4 () {};								
			
			//@}
		   
			//! \name Vector Accessing and Copying
			//@{
		   
			//! Make this vector a copy of the vector <i>v</i>
			void copy (const vec4 &v);

			//! Return a new vector with the same contents a this vector
			vec4 * clone () const;

			//! Initialize ou change the vector's components
			void set (float x,float y, float z, float w = 1.0);					
		    void set(vec4* aVec );
			void set(float *values);
			void set(const vec4& aVec );
			// Assignment operator
			const vec4& operator = (const vec4 &v);
		   
			//@}
		   
			//! \name Overloaded Arithmetic Operators
			//@{							
		   
			//! Vector addition
			const vec4& operator += (const vec4 &v); 		
		   
			//! Vector subtraction
			const vec4& operator -= (const vec4 &v);
		   
			//! Vector negation
			const vec4& operator - (void);				

			//! Vector scaling
			const vec4& operator *= (float t);
		   
			//! Scalar division
			const vec4& operator /= (float t);					
		   
			//@}
		   
		   
			//! \name Overloaded comparation
			//@{
			
			//! Equal
			bool operator == (const vec4 &v) const;
			
			//! Diferent
			bool operator != (const vec4 &v) const;

			//@}

			//! \name Methods
			//@{
		   
			//! Length of vector
			float length () const;
		   
			//! Length without the square root
			float sqrLength () const;					       
		   
			//! distance between this vector and vector <i>v</i>
			float distance (const vec4 &v) const;		
			
			//! Normalize this vector
			void normalize ();
		   
			//! Return the unit vector of this vector
			const vec4 unitVector () const;					
		   
			//! Dot product between this vector and vector <i>v</i>
			float dot (const vec4 &v) const;	
		   
			//! Cross product between this vector and vector <i>v</i>. 
			//! Assumes that what we really wnat is a 3D vector in homogeneous coordinates
			const vec4 cross (const vec4 &v) const;
		   
			//! Angle between two vectors.
			//float angle (vec4 &v) const;
		   
			//! Interpolated vector between this vector and vector <i>v</i> at
			//! position alpha (0.0 <= alpha <= 1.0)
			const vec4 lerp (const vec4 &v, float alpha) const;   
		   
			//! Add another vector to this one
			void add (const vec4 &v);
		   
			//! Scalar multiplication
			void scale (float a);				
		   
			//! Vector Equality
			bool equals (const vec4 &v, float tolerance=-1.0f) const;
		   
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
			//	ar & w;
			//}
			//END: boost serialization interface
		};
	};
};
#endif // vec4_H
