#ifndef VEC2_H
#define VEC2_H

//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/binary_iarchive.hpp>

namespace nau
{
	namespace math
	{

		//! \brief This class defines a vector in 2D space.
		//!
		//! \author Eduardo Marques
		//!	
		//! A vector defines both a magnitude (length) and a
		//! direction. One use for the vec2 class is as a data type for the
		//! entities in physics equations. A vector can also be use to
		//! define a point in 2D space ( a point is essentially a vector
		//! that originates at (0,0) ).
		//!		
		//! vec2 defines the three vector components (x,y) as data
		//! members and has several vector operations as member functions.
		class vec2 {
			//friend class boost::serialization::access;
		 public:
		  
		  /// Static copies of the unit vectors
		  static const vec2 
				UNIT_X, 
				UNIT_Y, 
				NEGATIVE_UNIT_X,
				NEGATIVE_UNIT_Y;
		  
			//! x, y, z vector components 
			float x,y;							
		   
			/// \name Constructors
			//@{
		   
			//! Default class constructor. Initialize vector to (0,0,0,0)
			explicit vec2 (): x(0.0f), y(0.0f) {};
		   
			//! Complete class constructor
			explicit vec2 (float x, float y): x(x), y(y) {};
		   
			//! Copy constructor. 
			vec2 (const vec2 &v);
		   
			//! Class destructor 
			~vec2 () {};								
			
			//@}
		   
			//! \name Vector Accessing and Copying
			//@{
		   
			//! Make this vector a copy of the vector <i>v</i>
			void copy (const vec2 &v);

			//! Return a new vector with the same contents a this vector
			vec2 * clone () const;

			//! Initialize ou change the vector's components
			void set (float x,float y);	

		   
			// Assignment operator
			const vec2& operator = (const vec2 &v);
		   
			//@}
		   
			//! \name Overloaded Arithmetic Operators
			//@{							
		   
			//! Vector addition
			const vec2& operator += (const vec2 &v); 		
		   
			//! Vector subtraction
			const vec2& operator -= (const vec2 &v);
		   
			//! Vector negation
			const vec2& operator - (void);				

			//! Vector scaling
			const vec2& operator *= (float t);
		   
			//! Scalar division
			const vec2& operator /= (float t);					
		   
			//@}
		   
		   
			//! \name Overloaded comparation
			//@{
			
			//! Equal
			bool operator == (const vec2 &v) const;
			
			//! Diferent
			bool operator != (const vec2 &v) const;

			//@}

			//! \name Methods
			//@{
		   
			//! Length of vector
			float length () const;
		   
			//! Length without the square root
			float sqrLength () const;					       
		   
			//! distance between this vector and vector <i>v</i>
			float distance (const vec2 &v) const;		
			
			//! Normalize this vector
			void normalize ();
		   
			//! Return the unit vector of this vector
			const vec2 unitVector () const;					
		   
			//! Dot product between this vector and vector <i>v</i>
			float dot (const vec2 &v) const;	
		   		   
			//! Angle between two vectors.
			float angle (vec2 &v) const;
		   
			//! Interpolated vector between this vector and vector <i>v</i> at
			//! position alpha (0.0 <= alpha <= 1.0)
			const vec2 lerp (const vec2 &v, float alpha) const;   
		   
			//! Add another vector to this one
			void add (const vec2 &v);
		   
			//! Scalar multiplication
			void scale (float a);				
		   
			//! Vector Equality
			bool equals (const vec2 &v, float tolerance=-1.0f) const;
		   
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
#endif // VEC2_H
