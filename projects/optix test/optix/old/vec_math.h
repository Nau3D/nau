//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


#pragma once



#include <vector_functions.h>
#include <vector_types.h>

#include <cmath>
#include <cstdlib>


/* scalar functions used in vector functions */
#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif


#if !defined(__CUDACC__)

inline   int max(int a, int b)
{
    return a > b ? a : b;
}

inline   int min(int a, int b)
{
    return a < b ? a : b;
}

inline   long long max(long long a, long long b)
{
    return a > b ? a : b;
}

inline   long long min(long long a, long long b)
{
    return a < b ? a : b;
}

inline   unsigned int max(unsigned int a, unsigned int b)
{
    return a > b ? a : b;
}

inline   unsigned int min(unsigned int a, unsigned int b)
{
    return a < b ? a : b;
}

inline   unsigned long long max(unsigned long long a, unsigned long long b)
{
    return a > b ? a : b;
}

inline   unsigned long long min(unsigned long long a, unsigned long long b)
{
    return a < b ? a : b;
}


/** lerp */
inline   float lerp(const float a, const float b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
inline   float bilerp(const float x00, const float x10, const float x01, const float x11,
                                         const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

template <typename IntegerType>
inline   IntegerType roundUp(IntegerType x, IntegerType y)
{
    return ( ( x + y - 1 ) / y ) * y;
}

#endif

/** clamp */
inline   float clamp(const float f, const float a, const float b)
{
  return fmaxf(a, fminf(f, b));
}


/* float2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
inline   float2 make_float2(const float s)
{
  return make_float2(s, s);
}
inline   float2 make_float2(const int2& a)
{
  return make_float2(float(a.x), float(a.y));
}
inline   float2 make_float2(const uint2& a)
{
  return make_float2(float(a.x), float(a.y));
}
/** @} */

/** negate */
inline   float2 operator-(const float2& a)
{
  return make_float2(-a.x, -a.y);
}

/** min 
* @{
*/
inline   float2 fminf(const float2& a, const float2& b)
{
  return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline   float fminf(const float2& a)
{
  return fminf(a.x, a.y);
}
/** @} */

/** max 
* @{
*/
inline   float2 fmaxf(const float2& a, const float2& b)
{
  return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline   float fmaxf(const float2& a)
{
  return fmaxf(a.x, a.y);
}
/** @} */

/** add 
* @{
*/
inline   float2 operator+(const float2& a, const float2& b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
inline   float2 operator+(const float2& a, const float b)
{
  return make_float2(a.x + b, a.y + b);
}
inline   float2 operator+(const float a, const float2& b)
{
  return make_float2(a + b.x, a + b.y);
}
inline   void operator+=(float2& a, const float2& b)
{
  a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract 
* @{
*/
inline   float2 operator-(const float2& a, const float2& b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}
inline   float2 operator-(const float2& a, const float b)
{
  return make_float2(a.x - b, a.y - b);
}
inline   float2 operator-(const float a, const float2& b)
{
  return make_float2(a - b.x, a - b.y);
}
inline   void operator-=(float2& a, const float2& b)
{
  a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply 
* @{
*/
inline   float2 operator*(const float2& a, const float2& b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}
inline   float2 operator*(const float2& a, const float s)
{
  return make_float2(a.x * s, a.y * s);
}
inline   float2 operator*(const float s, const float2& a)
{
  return make_float2(a.x * s, a.y * s);
}
inline   void operator*=(float2& a, const float2& s)
{
  a.x *= s.x; a.y *= s.y;
}
inline   void operator*=(float2& a, const float s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** divide 
* @{
*/
inline   float2 operator/(const float2& a, const float2& b)
{
  return make_float2(a.x / b.x, a.y / b.y);
}
inline   float2 operator/(const float2& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
inline   float2 operator/(const float s, const float2& a)
{
  return make_float2( s/a.x, s/a.y );
}
inline   void operator/=(float2& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
inline   float2 lerp(const float2& a, const float2& b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
inline   float2 bilerp(const float2& x00, const float2& x10, const float2& x01, const float2& x11,
                                          const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp 
* @{
*/
inline   float2 clamp(const float2& v, const float a, const float b)
{
  return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline   float2 clamp(const float2& v, const float2& a, const float2& b)
{
  return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** dot product */
inline   float dot(const float2& a, const float2& b)
{
  return a.x * b.x + a.y * b.y;
}

/** length */
inline   float length(const float2& v)
{
  return sqrtf(dot(v, v));
}

/** normalize */
inline   float2 normalize(const float2& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
inline   float2 floor(const float2& v)
{
  return make_float2(::floorf(v.x), ::floorf(v.y));
}

/** reflect */
inline   float2 reflect(const float2& i, const float2& n)
{
  return i - 2.0f * n * dot(n,i);
}

/** Faceforward
* Returns N if dot(i, nref) > 0; else -N; 
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL */
inline   float2 faceforward(const float2& n, const float2& i, const float2& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/** exp */
inline   float2 expf(const float2& v)
{
  return make_float2(::expf(v.x), ::expf(v.y));
}

/** If used on the device, this could place the the 'v' in local memory */
inline   float getByIndex(const float2& v, int i)
{
  return ((float*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(float2& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}


/* float3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
inline   float3 make_float3(const float s)
{
  return make_float3(s, s, s);
}
inline   float3 make_float3(const float2& a)
{
  return make_float3(a.x, a.y, 0.0f);
}
inline   float3 make_float3(const int3& a)
{
  return make_float3(float(a.x), float(a.y), float(a.z));
}
inline   float3 make_float3(const uint3& a)
{
  return make_float3(float(a.x), float(a.y), float(a.z));
}
/** @} */

/** negate */
inline   float3 operator-(const float3& a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

/** min 
* @{
*/
inline   float3 fminf(const float3& a, const float3& b)
{
  return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline   float fminf(const float3& a)
{
  return fminf(fminf(a.x, a.y), a.z);
}
/** @} */

/** max 
* @{
*/
inline   float3 fmaxf(const float3& a, const float3& b)
{
  return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline   float fmaxf(const float3& a)
{
  return fmaxf(fmaxf(a.x, a.y), a.z);
}
/** @} */

/** add 
* @{
*/
inline   float3 operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline   float3 operator+(const float3& a, const float b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}
inline   float3 operator+(const float a, const float3& b)
{
  return make_float3(a + b.x, a + b.y, a + b.z);
}
inline   void operator+=(float3& a, const float3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract 
* @{
*/
inline   float3 operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline   float3 operator-(const float3& a, const float b)
{
  return make_float3(a.x - b, a.y - b, a.z - b);
}
inline   float3 operator-(const float a, const float3& b)
{
  return make_float3(a - b.x, a - b.y, a - b.z);
}
inline   void operator-=(float3& a, const float3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply 
* @{
*/
inline   float3 operator*(const float3& a, const float3& b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline   float3 operator*(const float3& a, const float s)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
inline   float3 operator*(const float s, const float3& a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
inline   void operator*=(float3& a, const float3& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
inline   void operator*=(float3& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide 
* @{
*/
inline   float3 operator/(const float3& a, const float3& b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline   float3 operator/(const float3& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
inline   float3 operator/(const float s, const float3& a)
{
  return make_float3( s/a.x, s/a.y, s/a.z );
}
inline   void operator/=(float3& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
inline   float3 lerp(const float3& a, const float3& b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
inline   float3 bilerp(const float3& x00, const float3& x10, const float3& x01, const float3& x11,
                                          const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp 
* @{
*/
inline   float3 clamp(const float3& v, const float a, const float b)
{
  return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline   float3 clamp(const float3& v, const float3& a, const float3& b)
{
  return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** dot product */
inline   float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** cross product */
inline   float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

/** length */
inline   float length(const float3& v)
{
  return sqrtf(dot(v, v));
}

/** normalize */
inline   float3 normalize(const float3& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
inline   float3 floor(const float3& v)
{
  return make_float3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z));
}

/** reflect */
inline   float3 reflect(const float3& i, const float3& n)
{
  return i - 2.0f * n * dot(n,i);
}

/** Faceforward
* Returns N if dot(i, nref) > 0; else -N;
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL */
inline   float3 faceforward(const float3& n, const float3& i, const float3& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/** exp */
inline   float3 expf(const float3& v)
{
  return make_float3(::expf(v.x), ::expf(v.y), ::expf(v.z));
}

/** If used on the device, this could place the the 'v' in local memory */
inline   float getByIndex(const float3& v, int i)
{
  return ((float*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(float3& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}
  
/* float4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
inline   float4 make_float4(const float s)
{
  return make_float4(s, s, s, s);
}
inline   float4 make_float4(const float3& a)
{
  return make_float4(a.x, a.y, a.z, 0.0f);
}
inline   float4 make_float4(const int4& a)
{
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline   float4 make_float4(const uint4& a)
{
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
/** @} */

/** negate */
inline   float4 operator-(const float4& a)
{
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

/** min 
* @{
*/
inline   float4 fminf(const float4& a, const float4& b)
{
  return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}
inline   float fminf(const float4& a)
{
  return fminf(fminf(a.x, a.y), fminf(a.z, a.w));
}
/** @} */

/** max 
* @{
*/
inline   float4 fmaxf(const float4& a, const float4& b)
{
  return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}
inline   float fmaxf(const float4& a)
{
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}
/** @} */

/** add 
* @{
*/
inline   float4 operator+(const float4& a, const float4& b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline   float4 operator+(const float4& a, const float b)
{
  return make_float4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline   float4 operator+(const float a, const float4& b)
{
  return make_float4(a + b.x, a + b.y, a + b.z,  a + b.w);
}
inline   void operator+=(float4& a, const float4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract 
* @{
*/
inline   float4 operator-(const float4& a, const float4& b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline   float4 operator-(const float4& a, const float b)
{
  return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline   float4 operator-(const float a, const float4& b)
{
  return make_float4(a - b.x, a - b.y, a - b.z,  a - b.w);
}
inline   void operator-=(float4& a, const float4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply 
* @{
*/
inline   float4 operator*(const float4& a, const float4& s)
{
  return make_float4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}
inline   float4 operator*(const float4& a, const float s)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   float4 operator*(const float s, const float4& a)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   void operator*=(float4& a, const float4& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z; a.w *= s.w;
}
inline   void operator*=(float4& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
inline   float4 operator/(const float4& a, const float4& b)
{
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline   float4 operator/(const float4& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
inline   float4 operator/(const float s, const float4& a)
{
  return make_float4( s/a.x, s/a.y, s/a.z, s/a.w );
}
inline   void operator/=(float4& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
inline   float4 lerp(const float4& a, const float4& b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
inline   float4 bilerp(const float4& x00, const float4& x10, const float4& x01, const float4& x11,
                                          const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp 
* @{
*/
inline   float4 clamp(const float4& v, const float a, const float b)
{
  return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline   float4 clamp(const float4& v, const float4& a, const float4& b)
{
  return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** dot product */
inline   float dot(const float4& a, const float4& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/** length */
inline   float length(const float4& r)
{
  return sqrtf(dot(r, r));
}

/** normalize */
inline   float4 normalize(const float4& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
inline   float4 floor(const float4& v)
{
  return make_float4(::floorf(v.x), ::floorf(v.y), ::floorf(v.z), ::floorf(v.w));
}

/** reflect */
inline   float4 reflect(const float4& i, const float4& n)
{
  return i - 2.0f * n * dot(n,i);
}

/** 
* Faceforward
* Returns N if dot(i, nref) > 0; else -N;
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL 
*/
inline   float4 faceforward(const float4& n, const float4& i, const float4& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/** exp */
inline   float4 expf(const float4& v)
{
  return make_float4(::expf(v.x), ::expf(v.y), ::expf(v.z), ::expf(v.w));
}

/** If used on the device, this could place the the 'v' in local memory */
inline   float getByIndex(const float4& v, int i)
{
  return ((float*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(float4& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}
  
  
/* int functions */
/******************************************************************************/

/** clamp */
inline   int clamp(const int f, const int a, const int b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
inline   int getByIndex(const int1& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(int1& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* int2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
inline   int2 make_int2(const int s)
{
  return make_int2(s, s);
}
inline   int2 make_int2(const float2& a)
{
  return make_int2(int(a.x), int(a.y));
}
/** @} */

/** negate */
inline   int2 operator-(const int2& a)
{
  return make_int2(-a.x, -a.y);
}

/** min */
inline   int2 min(const int2& a, const int2& b)
{
  return make_int2(min(a.x,b.x), min(a.y,b.y));
}

/** max */
inline   int2 max(const int2& a, const int2& b)
{
  return make_int2(max(a.x,b.x), max(a.y,b.y));
}

/** add 
* @{
*/
inline   int2 operator+(const int2& a, const int2& b)
{
  return make_int2(a.x + b.x, a.y + b.y);
}
inline   void operator+=(int2& a, const int2& b)
{
  a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract 
* @{
*/
inline   int2 operator-(const int2& a, const int2& b)
{
  return make_int2(a.x - b.x, a.y - b.y);
}
inline   int2 operator-(const int2& a, const int b)
{
  return make_int2(a.x - b, a.y - b);
}
inline   void operator-=(int2& a, const int2& b)
{
  a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply 
* @{
*/
inline   int2 operator*(const int2& a, const int2& b)
{
  return make_int2(a.x * b.x, a.y * b.y);
}
inline   int2 operator*(const int2& a, const int s)
{
  return make_int2(a.x * s, a.y * s);
}
inline   int2 operator*(const int s, const int2& a)
{
  return make_int2(a.x * s, a.y * s);
}
inline   void operator*=(int2& a, const int s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** clamp 
* @{
*/
inline   int2 clamp(const int2& v, const int a, const int b)
{
  return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline   int2 clamp(const int2& v, const int2& a, const int2& b)
{
  return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality 
* @{
*/
inline   bool operator==(const int2& a, const int2& b)
{
  return a.x == b.x && a.y == b.y;
}

inline   bool operator!=(const int2& a, const int2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
inline   int getByIndex(const int2& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(int2& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* int3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
inline   int3 make_int3(const int s)
{
  return make_int3(s, s, s);
}
inline   int3 make_int3(const float3& a)
{
  return make_int3(int(a.x), int(a.y), int(a.z));
}
/** @} */

/** negate */
inline   int3 operator-(const int3& a)
{
  return make_int3(-a.x, -a.y, -a.z);
}

/** min */
inline   int3 min(const int3& a, const int3& b)
{
  return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

/** max */
inline   int3 max(const int3& a, const int3& b)
{
  return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

/** add 
* @{
*/
inline   int3 operator+(const int3& a, const int3& b)
{
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline   void operator+=(int3& a, const int3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract 
* @{
*/
inline   int3 operator-(const int3& a, const int3& b)
{
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline   void operator-=(int3& a, const int3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply 
* @{
*/
inline   int3 operator*(const int3& a, const int3& b)
{
  return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline   int3 operator*(const int3& a, const int s)
{
  return make_int3(a.x * s, a.y * s, a.z * s);
}
inline   int3 operator*(const int s, const int3& a)
{
  return make_int3(a.x * s, a.y * s, a.z * s);
}
inline   void operator*=(int3& a, const int s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide 
* @{
*/
inline   int3 operator/(const int3& a, const int3& b)
{
  return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline   int3 operator/(const int3& a, const int s)
{
  return make_int3(a.x / s, a.y / s, a.z / s);
}
inline   int3 operator/(const int s, const int3& a)
{
  return make_int3(s /a.x, s / a.y, s / a.z);
}
inline   void operator/=(int3& a, const int s)
{
  a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp 
* @{
*/
inline   int3 clamp(const int3& v, const int a, const int b)
{
  return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline   int3 clamp(const int3& v, const int3& a, const int3& b)
{
  return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality 
* @{
*/
inline   bool operator==(const int3& a, const int3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline   bool operator!=(const int3& a, const int3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
inline   int getByIndex(const int3& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(int3& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* int4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
inline   int4 make_int4(const int s)
{
  return make_int4(s, s, s, s);
}
inline   int4 make_int4(const float4& a)
{
  return make_int4((int)a.x, (int)a.y, (int)a.z, (int)a.w);
}
/** @} */

/** negate */
inline   int4 operator-(const int4& a)
{
  return make_int4(-a.x, -a.y, -a.z, -a.w);
}

/** min */
inline   int4 min(const int4& a, const int4& b)
{
  return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

/** max */
inline   int4 max(const int4& a, const int4& b)
{
  return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

/** add 
* @{
*/
inline   int4 operator+(const int4& a, const int4& b)
{
  return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline   void operator+=(int4& a, const int4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract 
* @{
*/
inline   int4 operator-(const int4& a, const int4& b)
{
  return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline   void operator-=(int4& a, const int4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply 
* @{
*/
inline   int4 operator*(const int4& a, const int4& b)
{
  return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline   int4 operator*(const int4& a, const int s)
{
  return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   int4 operator*(const int s, const int4& a)
{
  return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   void operator*=(int4& a, const int s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
inline   int4 operator/(const int4& a, const int4& b)
{
  return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline   int4 operator/(const int4& a, const int s)
{
  return make_int4(a.x / s, a.y / s, a.z / s, a.w / s);
}
inline   int4 operator/(const int s, const int4& a)
{
  return make_int4(s / a.x, s / a.y, s / a.z, s / a.w);
}
inline   void operator/=(int4& a, const int s)
{
  a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp 
* @{
*/
inline   int4 clamp(const int4& v, const int a, const int b)
{
  return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline   int4 clamp(const int4& v, const int4& a, const int4& b)
{
  return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality 
* @{
*/
inline   bool operator==(const int4& a, const int4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline   bool operator!=(const int4& a, const int4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
inline   int getByIndex(const int4& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(int4& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* uint functions */
/******************************************************************************/

/** clamp */
inline   unsigned int clamp(const unsigned int f, const unsigned int a, const unsigned int b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
inline   unsigned int getByIndex(const uint1& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(uint1& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}
  

/* uint2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
inline   uint2 make_uint2(const unsigned int s)
{
  return make_uint2(s, s);
}
inline   uint2 make_uint2(const float2& a)
{
  return make_uint2((unsigned int)a.x, (unsigned int)a.y);
}
/** @} */

/** min */
inline   uint2 min(const uint2& a, const uint2& b)
{
  return make_uint2(min(a.x,b.x), min(a.y,b.y));
}

/** max */
inline   uint2 max(const uint2& a, const uint2& b)
{
  return make_uint2(max(a.x,b.x), max(a.y,b.y));
}

/** add
* @{
*/
inline   uint2 operator+(const uint2& a, const uint2& b)
{
  return make_uint2(a.x + b.x, a.y + b.y);
}
inline   void operator+=(uint2& a, const uint2& b)
{
  a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
inline   uint2 operator-(const uint2& a, const uint2& b)
{
  return make_uint2(a.x - b.x, a.y - b.y);
}
inline   uint2 operator-(const uint2& a, const unsigned int b)
{
  return make_uint2(a.x - b, a.y - b);
}
inline   void operator-=(uint2& a, const uint2& b)
{
  a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
inline   uint2 operator*(const uint2& a, const uint2& b)
{
  return make_uint2(a.x * b.x, a.y * b.y);
}
inline   uint2 operator*(const uint2& a, const unsigned int s)
{
  return make_uint2(a.x * s, a.y * s);
}
inline   uint2 operator*(const unsigned int s, const uint2& a)
{
  return make_uint2(a.x * s, a.y * s);
}
inline   void operator*=(uint2& a, const unsigned int s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** clamp
* @{
*/
inline   uint2 clamp(const uint2& v, const unsigned int a, const unsigned int b)
{
  return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline   uint2 clamp(const uint2& v, const uint2& a, const uint2& b)
{
  return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
inline   bool operator==(const uint2& a, const uint2& b)
{
  return a.x == b.x && a.y == b.y;
}

inline   bool operator!=(const uint2& a, const uint2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
inline   unsigned int getByIndex(const uint2& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(uint2& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}
  

/* uint3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
inline   uint3 make_uint3(const unsigned int s)
{
  return make_uint3(s, s, s);
}
inline   uint3 make_uint3(const float3& a)
{
  return make_uint3((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z);
}
/** @} */

/** min */
inline   uint3 min(const uint3& a, const uint3& b)
{
  return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

/** max */
inline   uint3 max(const uint3& a, const uint3& b)
{
  return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

/** add 
* @{
*/
inline   uint3 operator+(const uint3& a, const uint3& b)
{
  return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline   void operator+=(uint3& a, const uint3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
inline   uint3 operator-(const uint3& a, const uint3& b)
{
  return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline   void operator-=(uint3& a, const uint3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
inline   uint3 operator*(const uint3& a, const uint3& b)
{
  return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline   uint3 operator*(const uint3& a, const unsigned int s)
{
  return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline   uint3 operator*(const unsigned int s, const uint3& a)
{
  return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline   void operator*=(uint3& a, const unsigned int s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
inline   uint3 operator/(const uint3& a, const uint3& b)
{
  return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline   uint3 operator/(const uint3& a, const unsigned int s)
{
  return make_uint3(a.x / s, a.y / s, a.z / s);
}
inline   uint3 operator/(const unsigned int s, const uint3& a)
{
  return make_uint3(s / a.x, s / a.y, s / a.z);
}
inline   void operator/=(uint3& a, const unsigned int s)
{
  a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp 
* @{
*/
inline   uint3 clamp(const uint3& v, const unsigned int a, const unsigned int b)
{
  return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline   uint3 clamp(const uint3& v, const uint3& a, const uint3& b)
{
  return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality 
* @{
*/
inline   bool operator==(const uint3& a, const uint3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline   bool operator!=(const uint3& a, const uint3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory 
*/
inline   unsigned int getByIndex(const uint3& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory 
*/
inline   void setByIndex(uint3& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}
  

/* uint4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
inline   uint4 make_uint4(const unsigned int s)
{
  return make_uint4(s, s, s, s);
}
inline   uint4 make_uint4(const float4& a)
{
  return make_uint4((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z, (unsigned int)a.w);
}
/** @} */

/** min
* @{
*/
inline   uint4 min(const uint4& a, const uint4& b)
{
  return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}
/** @} */

/** max 
* @{
*/
inline   uint4 max(const uint4& a, const uint4& b)
{
  return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}
/** @} */

/** add
* @{
*/
inline   uint4 operator+(const uint4& a, const uint4& b)
{
  return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline   void operator+=(uint4& a, const uint4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract 
* @{
*/
inline   uint4 operator-(const uint4& a, const uint4& b)
{
  return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline   void operator-=(uint4& a, const uint4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
inline   uint4 operator*(const uint4& a, const uint4& b)
{
  return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline   uint4 operator*(const uint4& a, const unsigned int s)
{
  return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   uint4 operator*(const unsigned int s, const uint4& a)
{
  return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   void operator*=(uint4& a, const unsigned int s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
inline   uint4 operator/(const uint4& a, const uint4& b)
{
  return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline   uint4 operator/(const uint4& a, const unsigned int s)
{
  return make_uint4(a.x / s, a.y / s, a.z / s, a.w / s);
}
inline   uint4 operator/(const unsigned int s, const uint4& a)
{
  return make_uint4(s / a.x, s / a.y, s / a.z, s / a.w);
}
inline   void operator/=(uint4& a, const unsigned int s)
{
  a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp 
* @{
*/
inline   uint4 clamp(const uint4& v, const unsigned int a, const unsigned int b)
{
  return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline   uint4 clamp(const uint4& v, const uint4& a, const uint4& b)
{
  return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality 
* @{
*/
inline   bool operator==(const uint4& a, const uint4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline   bool operator!=(const uint4& a, const uint4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory 
*/
inline   unsigned int getByIndex(const uint4& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory 
*/
inline   void setByIndex(uint4& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}

/* long long functions */
/******************************************************************************/

/** clamp */
inline   long long clamp(const long long f, const long long a, const long long b)
{
    return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
inline   long long getByIndex(const longlong1& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(longlong1& v, int i, long long x)
{
    ((long long*)(&v))[i] = x;
}


/* longlong2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
inline   longlong2 make_longlong2(const long long s)
{
    return make_longlong2(s, s);
}
inline   longlong2 make_longlong2(const float2& a)
{
    return make_longlong2(int(a.x), int(a.y));
}
/** @} */

/** negate */
inline   longlong2 operator-(const longlong2& a)
{
    return make_longlong2(-a.x, -a.y);
}

/** min */
inline   longlong2 min(const longlong2& a, const longlong2& b)
{
    return make_longlong2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
inline   longlong2 max(const longlong2& a, const longlong2& b)
{
    return make_longlong2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
inline   longlong2 operator+(const longlong2& a, const longlong2& b)
{
    return make_longlong2(a.x + b.x, a.y + b.y);
}
inline   void operator+=(longlong2& a, const longlong2& b)
{
    a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
inline   longlong2 operator-(const longlong2& a, const longlong2& b)
{
    return make_longlong2(a.x - b.x, a.y - b.y);
}
inline   longlong2 operator-(const longlong2& a, const long long b)
{
    return make_longlong2(a.x - b, a.y - b);
}
inline   void operator-=(longlong2& a, const longlong2& b)
{
    a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
inline   longlong2 operator*(const longlong2& a, const longlong2& b)
{
    return make_longlong2(a.x * b.x, a.y * b.y);
}
inline   longlong2 operator*(const longlong2& a, const long long s)
{
    return make_longlong2(a.x * s, a.y * s);
}
inline   longlong2 operator*(const long long s, const longlong2& a)
{
    return make_longlong2(a.x * s, a.y * s);
}
inline   void operator*=(longlong2& a, const long long s)
{
    a.x *= s; a.y *= s;
}
/** @} */

/** clamp
* @{
*/
inline   longlong2 clamp(const longlong2& v, const long long a, const long long b)
{
    return make_longlong2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline   longlong2 clamp(const longlong2& v, const longlong2& a, const longlong2& b)
{
    return make_longlong2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
inline   bool operator==(const longlong2& a, const longlong2& b)
{
    return a.x == b.x && a.y == b.y;
}

inline   bool operator!=(const longlong2& a, const longlong2& b)
{
    return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
inline   long long getByIndex(const longlong2& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(longlong2& v, int i, long long x)
{
    ((long long*)(&v))[i] = x;
}


/* longlong3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
inline   longlong3 make_longlong3(const long long s)
{
    return make_longlong3(s, s, s);
}
inline   longlong3 make_longlong3(const float3& a)
{
    return make_longlong3( (long long)a.x, (long long)a.y, (long long)a.z);
}
/** @} */

/** negate */
inline   longlong3 operator-(const longlong3& a)
{
    return make_longlong3(-a.x, -a.y, -a.z);
}

/** min */
inline   longlong3 min(const longlong3& a, const longlong3& b)
{
    return make_longlong3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
inline   longlong3 max(const longlong3& a, const longlong3& b)
{
    return make_longlong3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
inline   longlong3 operator+(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline   void operator+=(longlong3& a, const longlong3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
inline   longlong3 operator-(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline   void operator-=(longlong3& a, const longlong3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
inline   longlong3 operator*(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline   longlong3 operator*(const longlong3& a, const long long s)
{
    return make_longlong3(a.x * s, a.y * s, a.z * s);
}
inline   longlong3 operator*(const long long s, const longlong3& a)
{
    return make_longlong3(a.x * s, a.y * s, a.z * s);
}
inline   void operator*=(longlong3& a, const long long s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
inline   longlong3 operator/(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline   longlong3 operator/(const longlong3& a, const long long s)
{
    return make_longlong3(a.x / s, a.y / s, a.z / s);
}
inline   longlong3 operator/(const long long s, const longlong3& a)
{
    return make_longlong3(s /a.x, s / a.y, s / a.z);
}
inline   void operator/=(longlong3& a, const long long s)
{
    a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp
* @{
*/
inline   longlong3 clamp(const longlong3& v, const long long a, const long long b)
{
    return make_longlong3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline   longlong3 clamp(const longlong3& v, const longlong3& a, const longlong3& b)
{
    return make_longlong3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
inline   bool operator==(const longlong3& a, const longlong3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline   bool operator!=(const longlong3& a, const longlong3& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
inline   long long getByIndex(const longlong3& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(longlong3& v, int i, int x)
{
    ((long long*)(&v))[i] = x;
}


/* longlong4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
inline   longlong4 make_longlong4(const long long s)
{
    return make_longlong4(s, s, s, s);
}
inline   longlong4 make_longlong4(const float4& a)
{
    return make_longlong4((long long)a.x, (long long)a.y, (long long)a.z, (long long)a.w);
}
/** @} */

/** negate */
inline   longlong4 operator-(const longlong4& a)
{
    return make_longlong4(-a.x, -a.y, -a.z, -a.w);
}

/** min */
inline   longlong4 min(const longlong4& a, const longlong4& b)
{
    return make_longlong4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

/** max */
inline   longlong4 max(const longlong4& a, const longlong4& b)
{
    return make_longlong4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

/** add
* @{
*/
inline   longlong4 operator+(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline   void operator+=(longlong4& a, const longlong4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
inline   longlong4 operator-(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline   void operator-=(longlong4& a, const longlong4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
inline   longlong4 operator*(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline   longlong4 operator*(const longlong4& a, const long long s)
{
    return make_longlong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   longlong4 operator*(const long long s, const longlong4& a)
{
    return make_longlong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   void operator*=(longlong4& a, const long long s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide
* @{
*/
inline   longlong4 operator/(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline   longlong4 operator/(const longlong4& a, const long long s)
{
    return make_longlong4(a.x / s, a.y / s, a.z / s, a.w / s);
}
inline   longlong4 operator/(const long long s, const longlong4& a)
{
    return make_longlong4(s / a.x, s / a.y, s / a.z, s / a.w);
}
inline   void operator/=(longlong4& a, const long long s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp
* @{
*/
inline   longlong4 clamp(const longlong4& v, const long long a, const long long b)
{
    return make_longlong4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline   longlong4 clamp(const longlong4& v, const longlong4& a, const longlong4& b)
{
    return make_longlong4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
inline   bool operator==(const longlong4& a, const longlong4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline   bool operator!=(const longlong4& a, const longlong4& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
inline   long long getByIndex(const longlong4& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(longlong4& v, int i, long long x)
{
    ((long long*)(&v))[i] = x;
}

/* ulonglong functions */
/******************************************************************************/

/** clamp */
inline   unsigned long long clamp(const unsigned long long f, const unsigned long long a, const unsigned long long b)
{
    return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
inline   unsigned long long getByIndex(const ulonglong1& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(ulonglong1& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
inline   ulonglong2 make_ulonglong2(const unsigned long long s)
{
    return make_ulonglong2(s, s);
}
inline   ulonglong2 make_ulonglong2(const float2& a)
{
    return make_ulonglong2((unsigned long long)a.x, (unsigned long long)a.y);
}
/** @} */

/** min */
inline   ulonglong2 min(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
inline   ulonglong2 max(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
inline   ulonglong2 operator+(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(a.x + b.x, a.y + b.y);
}
inline   void operator+=(ulonglong2& a, const ulonglong2& b)
{
    a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
inline   ulonglong2 operator-(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(a.x - b.x, a.y - b.y);
}
inline   ulonglong2 operator-(const ulonglong2& a, const unsigned long long b)
{
    return make_ulonglong2(a.x - b, a.y - b);
}
inline   void operator-=(ulonglong2& a, const ulonglong2& b)
{
    a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
inline   ulonglong2 operator*(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(a.x * b.x, a.y * b.y);
}
inline   ulonglong2 operator*(const ulonglong2& a, const unsigned long long s)
{
    return make_ulonglong2(a.x * s, a.y * s);
}
inline   ulonglong2 operator*(const unsigned long long s, const ulonglong2& a)
{
    return make_ulonglong2(a.x * s, a.y * s);
}
inline   void operator*=(ulonglong2& a, const unsigned long long s)
{
    a.x *= s; a.y *= s;
}
/** @} */

/** clamp
* @{
*/
inline   ulonglong2 clamp(const ulonglong2& v, const unsigned long long a, const unsigned long long b)
{
    return make_ulonglong2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline   ulonglong2 clamp(const ulonglong2& v, const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
inline   bool operator==(const ulonglong2& a, const ulonglong2& b)
{
    return a.x == b.x && a.y == b.y;
}

inline   bool operator!=(const ulonglong2& a, const ulonglong2& b)
{
    return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
inline   unsigned long long getByIndex(const ulonglong2& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
inline   void setByIndex(ulonglong2& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
inline   ulonglong3 make_ulonglong3(const unsigned long long s)
{
    return make_ulonglong3(s, s, s);
}
inline   ulonglong3 make_ulonglong3(const float3& a)
{
    return make_ulonglong3((unsigned long long)a.x, (unsigned long long)a.y, (unsigned long long)a.z);
}
/** @} */

/** min */
inline   ulonglong3 min(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
inline   ulonglong3 max(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
inline   ulonglong3 operator+(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline   void operator+=(ulonglong3& a, const ulonglong3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
inline   ulonglong3 operator-(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline   void operator-=(ulonglong3& a, const ulonglong3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
inline   ulonglong3 operator*(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline   ulonglong3 operator*(const ulonglong3& a, const unsigned long long s)
{
    return make_ulonglong3(a.x * s, a.y * s, a.z * s);
}
inline   ulonglong3 operator*(const unsigned long long s, const ulonglong3& a)
{
    return make_ulonglong3(a.x * s, a.y * s, a.z * s);
}
inline   void operator*=(ulonglong3& a, const unsigned long long s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
inline   ulonglong3 operator/(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline   ulonglong3 operator/(const ulonglong3& a, const unsigned long long s)
{
    return make_ulonglong3(a.x / s, a.y / s, a.z / s);
}
inline   ulonglong3 operator/(const unsigned long long s, const ulonglong3& a)
{
    return make_ulonglong3(s / a.x, s / a.y, s / a.z);
}
inline   void operator/=(ulonglong3& a, const unsigned long long s)
{
    a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp
* @{
*/
inline   ulonglong3 clamp(const ulonglong3& v, const unsigned long long a, const unsigned long long b)
{
    return make_ulonglong3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline   ulonglong3 clamp(const ulonglong3& v, const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
inline   bool operator==(const ulonglong3& a, const ulonglong3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline   bool operator!=(const ulonglong3& a, const ulonglong3& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
inline   unsigned long long getByIndex(const ulonglong3& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
*/
inline   void setByIndex(ulonglong3& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
inline   ulonglong4 make_ulonglong4(const unsigned long long s)
{
    return make_ulonglong4(s, s, s, s);
}
inline   ulonglong4 make_ulonglong4(const float4& a)
{
    return make_ulonglong4((unsigned long long)a.x, (unsigned long long)a.y, (unsigned long long)a.z, (unsigned long long)a.w);
}
/** @} */

/** min
* @{
*/
inline   ulonglong4 min(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}
/** @} */

/** max
* @{
*/
inline   ulonglong4 max(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}
/** @} */

/** add
* @{
*/
inline   ulonglong4 operator+(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline   void operator+=(ulonglong4& a, const ulonglong4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
inline   ulonglong4 operator-(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline   void operator-=(ulonglong4& a, const ulonglong4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
inline   ulonglong4 operator*(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline   ulonglong4 operator*(const ulonglong4& a, const unsigned long long s)
{
    return make_ulonglong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   ulonglong4 operator*(const unsigned long long s, const ulonglong4& a)
{
    return make_ulonglong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline   void operator*=(ulonglong4& a, const unsigned long long s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide
* @{
*/
inline   ulonglong4 operator/(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline   ulonglong4 operator/(const ulonglong4& a, const unsigned long long s)
{
    return make_ulonglong4(a.x / s, a.y / s, a.z / s, a.w / s);
}
inline   ulonglong4 operator/(const unsigned long long s, const ulonglong4& a)
{
    return make_ulonglong4(s / a.x, s / a.y, s / a.z, s / a.w);
}
inline   void operator/=(ulonglong4& a, const unsigned long long s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp
* @{
*/
inline   ulonglong4 clamp(const ulonglong4& v, const unsigned long long a, const unsigned long long b)
{
    return make_ulonglong4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline   ulonglong4 clamp(const ulonglong4& v, const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
inline   bool operator==(const ulonglong4& a, const ulonglong4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline   bool operator!=(const ulonglong4& a, const ulonglong4& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
inline   unsigned long long getByIndex(const ulonglong4& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
*/
inline   void setByIndex(ulonglong4& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/******************************************************************************/

/** Narrowing functions
* @{
*/
inline   int2 make_int2(const int3& v0) { return make_int2( v0.x, v0.y ); }
inline   int2 make_int2(const int4& v0) { return make_int2( v0.x, v0.y ); }
inline   int3 make_int3(const int4& v0) { return make_int3( v0.x, v0.y, v0.z ); }
inline   uint2 make_uint2(const uint3& v0) { return make_uint2( v0.x, v0.y ); }
inline   uint2 make_uint2(const uint4& v0) { return make_uint2( v0.x, v0.y ); }
inline   uint3 make_uint3(const uint4& v0) { return make_uint3( v0.x, v0.y, v0.z ); }
inline   longlong2 make_longlong2(const longlong3& v0) { return make_longlong2( v0.x, v0.y ); }
inline   longlong2 make_longlong2(const longlong4& v0) { return make_longlong2( v0.x, v0.y ); }
inline   longlong3 make_longlong3(const longlong4& v0) { return make_longlong3( v0.x, v0.y, v0.z ); }
inline   ulonglong2 make_ulonglong2(const ulonglong3& v0) { return make_ulonglong2( v0.x, v0.y ); }
inline   ulonglong2 make_ulonglong2(const ulonglong4& v0) { return make_ulonglong2( v0.x, v0.y ); }
inline   ulonglong3 make_ulonglong3(const ulonglong4& v0) { return make_ulonglong3( v0.x, v0.y, v0.z ); }
inline   float2 make_float2(const float3& v0) { return make_float2( v0.x, v0.y ); }
inline   float2 make_float2(const float4& v0) { return make_float2( v0.x, v0.y ); }
inline   float3 make_float3(const float4& v0) { return make_float3( v0.x, v0.y, v0.z ); }
/** @} */

/** Assemble functions from smaller vectors 
* @{
*/
inline   int3 make_int3(const int v0, const int2& v1) { return make_int3( v0, v1.x, v1.y ); }
inline   int3 make_int3(const int2& v0, const int v1) { return make_int3( v0.x, v0.y, v1 ); }
inline   int4 make_int4(const int v0, const int v1, const int2& v2) { return make_int4( v0, v1, v2.x, v2.y ); }
inline   int4 make_int4(const int v0, const int2& v1, const int v2) { return make_int4( v0, v1.x, v1.y, v2 ); }
inline   int4 make_int4(const int2& v0, const int v1, const int v2) { return make_int4( v0.x, v0.y, v1, v2 ); }
inline   int4 make_int4(const int v0, const int3& v1) { return make_int4( v0, v1.x, v1.y, v1.z ); }
inline   int4 make_int4(const int3& v0, const int v1) { return make_int4( v0.x, v0.y, v0.z, v1 ); }
inline   int4 make_int4(const int2& v0, const int2& v1) { return make_int4( v0.x, v0.y, v1.x, v1.y ); }
inline   uint3 make_uint3(const unsigned int v0, const uint2& v1) { return make_uint3( v0, v1.x, v1.y ); }
inline   uint3 make_uint3(const uint2& v0, const unsigned int v1) { return make_uint3( v0.x, v0.y, v1 ); }
inline   uint4 make_uint4(const unsigned int v0, const unsigned int v1, const uint2& v2) { return make_uint4( v0, v1, v2.x, v2.y ); }
inline   uint4 make_uint4(const unsigned int v0, const uint2& v1, const unsigned int v2) { return make_uint4( v0, v1.x, v1.y, v2 ); }
inline   uint4 make_uint4(const uint2& v0, const unsigned int v1, const unsigned int v2) { return make_uint4( v0.x, v0.y, v1, v2 ); }
inline   uint4 make_uint4(const unsigned int v0, const uint3& v1) { return make_uint4( v0, v1.x, v1.y, v1.z ); }
inline   uint4 make_uint4(const uint3& v0, const unsigned int v1) { return make_uint4( v0.x, v0.y, v0.z, v1 ); }
inline   uint4 make_uint4(const uint2& v0, const uint2& v1) { return make_uint4( v0.x, v0.y, v1.x, v1.y ); }
inline   longlong3 make_longlong3(const long long v0, const longlong2& v1) { return make_longlong3(v0, v1.x, v1.y); }
inline   longlong3 make_longlong3(const longlong2& v0, const long long v1) { return make_longlong3(v0.x, v0.y, v1); }
inline   longlong4 make_longlong4(const long long v0, const long long v1, const longlong2& v2) { return make_longlong4(v0, v1, v2.x, v2.y); }
inline   longlong4 make_longlong4(const long long v0, const longlong2& v1, const long long v2) { return make_longlong4(v0, v1.x, v1.y, v2); }
inline   longlong4 make_longlong4(const longlong2& v0, const long long v1, const long long v2) { return make_longlong4(v0.x, v0.y, v1, v2); }
inline   longlong4 make_longlong4(const long long v0, const longlong3& v1) { return make_longlong4(v0, v1.x, v1.y, v1.z); }
inline   longlong4 make_longlong4(const longlong3& v0, const long long v1) { return make_longlong4(v0.x, v0.y, v0.z, v1); }
inline   longlong4 make_longlong4(const longlong2& v0, const longlong2& v1) { return make_longlong4(v0.x, v0.y, v1.x, v1.y); }
inline   ulonglong3 make_ulonglong3(const unsigned long long v0, const ulonglong2& v1) { return make_ulonglong3(v0, v1.x, v1.y); }
inline   ulonglong3 make_ulonglong3(const ulonglong2& v0, const unsigned long long v1) { return make_ulonglong3(v0.x, v0.y, v1); }
inline   ulonglong4 make_ulonglong4(const unsigned long long v0, const unsigned long long v1, const ulonglong2& v2) { return make_ulonglong4(v0, v1, v2.x, v2.y); }
inline   ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong2& v1, const unsigned long long v2) { return make_ulonglong4(v0, v1.x, v1.y, v2); }
inline   ulonglong4 make_ulonglong4(const ulonglong2& v0, const unsigned long long v1, const unsigned long long v2) { return make_ulonglong4(v0.x, v0.y, v1, v2); }
inline   ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong3& v1) { return make_ulonglong4(v0, v1.x, v1.y, v1.z); }
inline   ulonglong4 make_ulonglong4(const ulonglong3& v0, const unsigned long long v1) { return make_ulonglong4(v0.x, v0.y, v0.z, v1); }
inline   ulonglong4 make_ulonglong4(const ulonglong2& v0, const ulonglong2& v1) { return make_ulonglong4(v0.x, v0.y, v1.x, v1.y); }
inline   float3 make_float3(const float2& v0, const float v1) { return make_float3(v0.x, v0.y, v1); }
inline   float3 make_float3(const float v0, const float2& v1) { return make_float3( v0, v1.x, v1.y ); }
inline   float4 make_float4(const float v0, const float v1, const float2& v2) { return make_float4( v0, v1, v2.x, v2.y ); }
inline   float4 make_float4(const float v0, const float2& v1, const float v2) { return make_float4( v0, v1.x, v1.y, v2 ); }
inline   float4 make_float4(const float2& v0, const float v1, const float v2) { return make_float4( v0.x, v0.y, v1, v2 ); }
inline   float4 make_float4(const float v0, const float3& v1) { return make_float4( v0, v1.x, v1.y, v1.z ); }
inline   float4 make_float4(const float3& v0, const float v1) { return make_float4( v0.x, v0.y, v0.z, v1 ); }
inline   float4 make_float4(const float2& v0, const float2& v1) { return make_float4( v0.x, v0.y, v1.x, v1.y ); }
/** @} */


