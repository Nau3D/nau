#ifndef ITRANSFORM_H
#define ITRANSFORM_H

#include <nau/math/mat4.h>
#include <nau/math/vec3.h>


#include <string>

using namespace nau::math;

namespace nau
{
	namespace math
	{
		//! \brief This interface encapsulates a geometrical transforms
		class ITransform
		{
		public:
		   
			virtual void clone(ITransform* s) = 0;
			virtual ITransform* clone() = 0;

			virtual bool isIdentity() = 0;

			//! \name Accessors
			//@{
		   
			//! Get the contents of the transform as a mat4 object
			virtual const mat4 &getMat44 () const = 0;
		   
			//! Set the transform from a mat4 object
			virtual void setMat44 (mat4 &m) = 0;
		   
			//! Set the translation portion of the transform from axis coordinates
			virtual void setTranslation (float x, float y, float z) = 0;
		   
			//! Set the translation portion of the transform from a vec3 object
			virtual void setTranslation (const vec3 &v) = 0;
		   	   
			//! Set the rotation portion of the individual axis and angle values
			virtual void setRotation (float angle, float x, float y, float z) = 0;
		   
			//! Set the rotation portion of the transform from an angle and a
			//! vec3 object representin the axis of rotation.
			virtual void setRotation (float angle, const vec3 &v) = 0;
		   
		   
			//! Set the uniform scale portion of the transform
			virtual void setScale (float amount) = 0;
		   
			//! Set the uniform scale portion of the transform
			virtual void setScale (vec3 &v) = 0;
		   
			////! Get the uniform scale portion of the transform
			//virtual float getScale () = 0;
			////! Get the rotation angle of the transform
			//virtual float getRotationAngle () = 0;
		 //  
			////! Get the rotation axis of the transform as a vec3 object
			//virtual const vec3 &getRotationAxis () = 0; 
			//! Get the translation portion of the transform as a vec3 object
			virtual const vec3 &getTranslation () = 0;

			//@}
		   
			//! \name Methods
			//@{
		   
			//! Apply a translation to the current transform
			virtual void translate (float x, float y, float z) = 0;
		   
			//! Apply a translation to the current transform 
			virtual void translate (const vec3 &v) = 0;
		   
			//! Apply a rotation to the current transform
			virtual void rotate (float angle, float x, float y, float z) = 0;
		   
			//! Apply a rotation to the current transform
			virtual void rotate (float angle, vec3 &v) = 0;
		   
			//! Apply a uniform scaling to the current transform  
			virtual void scale (float amount) = 0;
		   
			//! Apply a non-uniform scale
			virtual void scale (vec3 &v) = 0;
			virtual void scale (float x, float y, float z) = 0;

			//! Compose the current transform with an arbitrary transform
			virtual void compose (const ITransform &t) = 0;
		   
			//! Inverts the current transform
			virtual void invert (void) = 0;

			//! Transpose the current matrix
			virtual void transpose() = 0;
			//@}

			virtual void setIdentity (void) = 0;

			virtual std::string getType (void) const = 0;

			//! Destructor
			virtual ~ITransform () {};
		};
	};
};
#endif // ITRANSFORM_H
