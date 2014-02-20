#ifndef SIMPLETRANSFORM_H
#define SIMPLETRANSFORM_H

#include <nau/math/itransform.h>


namespace nau
{
	namespace math
	{

		//! \brief This class is a simple implementation of the ITransform
		//! interface 
		//!
		//! \sa ITransform
		class SimpleTransform : public ITransform {

		 private:
			 mat4 m_Matrix;
			 vec3 m_Translation;
			 vec3 m_RotAxis;
			 vec3 m_Angle;
			 vec3 m_Scale;
		   
		 public:

			//! Constructor
			 SimpleTransform();
		   
			//! Destructor
			~SimpleTransform ();
		   
			void clone(ITransform *s);
			virtual ITransform* clone();


			bool isIdentity();
			//! \name Accessors
			//@{
		   
			//! Get the contents of the transform as a mat4 object
			const mat4 & getMat44 () const;
		   
			//! Set the transform from a mat4 object
			void setMat44 (mat4 &m);
		   
			//! Set the translation portion of the transform from axis coordinates
			void setTranslation (float x, float y, float z);
		   
			//! Set the translation portion of the transform from a vec3 object
			void setTranslation (const vec3 &v);
		   
			//! Set the rotation portion of the individual axis and angle values
			void setRotation (float angle, float x, float y, float z);
		   
		   
			//! Set a uniform scale 
			void setScale (float amount);
		   		   
			//! Set a non-uniform scale 
			void setScale (vec3 &v);
		   		   
			//! Set the rotation from an angle and a
			//! vec3 object representin the axis of rotation.
			void setRotation (float angle, const vec3 &v);
		   
			////! Get the rotation angle of the transform
			//float getRotationAngle ();		   
			////! Get the rotation axis of the transform as a vec3 object
			//const vec3 & getRotationAxis (); 
			//! Get the translation portion of the transform as a vec3 object
			const vec3 & getTranslation ();
			////! Get the uniform scale portion of the transform
			//float getScale ();

		   
			//@}
		   
			//! \name Methods
			//@{
		   
			//! Apply a translation to the current transform
			void translate (float x, float y, float z);
		   
			//! Apply a translation to the current transform 
			void translate (const vec3 &v);
		   
			//! Apply a rotation to the current transform
			void rotate (float angle, float x, float y, float z);
		   
			//! Apply a rotation to the current transform
			void rotate (float angle, vec3 &v);
		   
			//! Apply a uniform scaling to the current transform  
			void scale (float amount);

			//! Apply a non-uniform scale
			void scale (vec3 &v);
			void scale (float x, float y, float z);
		   
			//! Compose the current transform with an arbitrary transform
			void compose (const ITransform &t);
		   
			//! Inverts the current transform
			void invert (void);

			//! Transpose the current matrix
			void transpose();

			void setIdentity (void);

			//@}

			std::string getType (void) const;

//		 private:
		  
			
			////! Revoked assignment operator
			//const SimpleTransform & operator = (const SimpleTransform &m);


		};
	};
};
#endif // SIMPLETRANSFORM_H
