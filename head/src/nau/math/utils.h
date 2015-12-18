#ifndef MATHUTILS_H
#define MATHUTILS_H


#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <cfloat>

namespace nau
{
  namespace math
  {

    //
    // Generically useful math functions 
    //
    
    // convert degrees to radians
    static inline float 
      DegToRad(float degrees) 
    { 
      return (float)(degrees * (M_PI / 180.0f));
    };
    
    // convert radians to degrees
    static inline float 
      RadToDeg(float radians) 
    { 
      return (float)(radians * (180.0f / M_PI));
    };
    
    // test floating point values for equality
    static inline bool 
		FloatEqual(float a, float b, float tolerance = FLT_EPSILON)
    {
      float tol = tolerance;
      if (tol < 0.0f) { tol = FLT_EPSILON; }
      if (fabs(a - b) > tol) { return false; }
      else { return true; }
    };
    
    // Minimum between two floating point values
    static inline float
    FloatMin (float a, float b) 
    {
      return (a < b ? a : b);
    }
    
    // Maximum between two floating point values
    static inline float
    FloatMax (float a, float b) 
    {
      return (a > b ? a : b);
    }
    
    // Clamp a floating point value to a range
    static inline float
    FloatClamp (float value, float min, float max) 
    {
      return (FloatMin (FloatMax (min, value), max));
    }
  };
};

#endif // MATHUTILS_H
