#ifndef bvec4_H
#define bvec4_H


namespace nau
{
	namespace math
	{

		//! \brief This class defines a bool vector in 4D space.
		//!
		class bvec4 {
			//friend class boost::serialization::access;
		 public:
		  
			//! x, y, z vector components 
			bool x,y,z,w;							
		   
			/// \name Constructors
			//@{
		   
			//! Default class constructor. Initialize vector to (0,0,0,0)
			explicit bvec4 (): x(true), y(true), z(true), w(true) {};
		   
			//! Complete class constructor
			explicit bvec4 (bool x, bool y, bool z, bool w): x(x), y(y), z(z), w(w) {};
		   
			//! Copy constructor. 
			bvec4 (const bvec4 &v);
		   
			//! Class destructor 
			~bvec4 () {};								
			
			//@}
		   
			//! \name Vector Accessing and Copying
			//@{
		   
			//! Make this vector a copy of the vector <i>v</i>
			void copy (const bvec4 &v);

			//! Return a new vector with the same contents a this vector
			bvec4 * clone () const;

			//! Initialize ou change the vector's components
			void set (bool x,bool y, bool z, bool w );					
		    void set(bvec4* aVec );
			void set(bool *values);
			void set(const bvec4& aVec );
			// Assignment operator
			const bvec4& operator = (const bvec4 &v);
		   
			//@}		   
		   
			//! \name Overloaded comparation
			//@{
			
			//! Equal
			bool operator == (const bvec4 &v) const;
			
			//! Diferent
			bool operator != (const bvec4 &v) const;

			//@}

		};
	};
};
#endif // bvec4_H
