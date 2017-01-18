// This file contains some utility routines





// Create ONB from normalalized vector
// from nvidia's toolkit
__device__ __inline__ void createONB( const optix::float3& n,
                                      optix::float3& U,
                                      optix::float3& V)
{
	using namespace optix;
	
	U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );
	if ( dot(U, U) < 1.e-3f )
	  U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );
	U = normalize( U );
	V = cross( U, n );
}


// from nvidia's toolkit
__host__ __device__ __inline__ optix::float3 sample_phong_lobe( optix::float2 sample, float exponent, 
                                                                optix::float3 U, optix::float3 V, optix::float3 W )
{
	const float power = expf( logf(sample.y)/(exponent+1.0f) );
	const float phi = sample.x * 2.0f * (float)M_PIf;
	const float scale = sqrtf(1.0f - power*power);
	
	const float x = cosf(phi)*scale;
	const float y = sinf(phi)*scale;
	const float z = power;
	
	return x*U + y*V + z*W;
}


__device__ void sampleUnitHemisphereCosLobe(float3& normal, float exponent, float3& point, unsigned int &seed)
{
	float2 r;
	float3 U,V;

	r.x = rnd(seed);
	r.y = rnd(seed);

	createONB(normal, U, V);
	
	point = sample_phong_lobe(r, exponent, V, U, normal);
}


// from nvidia's toolkit
__device__ void mapToDisk( optix::float2& sample )
{
  float phi, r;
  float a = 2.0f * sample.x - 1.0f;      // (a,b) is now on [-1,1]^2 
  float b = 2.0f * sample.y - 1.0f;      // 
  if (a > -b) {                           // reg 1 or 2 
    if (a > b) {                          // reg 1, also |a| > |b| 
      r = a;
      phi = (M_PIf/4.0f) * (b/a);
    } else {                              // reg 2, also |b| > |a| 
      r = b;
      phi = (M_PIf/4.0f) * (2.0f - a/b);
    }
  } else {                                // reg 3 or 4 
    if (a < b) {                          // reg 3, also |a|>=|b| && a!=0 
      r = -a;
      phi = (M_PIf/4.0f) * (4.0f + b/a);
    } else {                              // region 4, |b| >= |a|,  but 
      // a==0 and  b==0 could occur. 
      r = -b;
      phi = b != 0.0f ? (M_PIf/4.0f) * (6.0f - a/b) :
        0.0f;
    }
  }
  float u = r * cosf( phi );
  float v = r * sinf( phi );
  sample.x = u;
  sample.y = v;
}


// sample hemisphere with cosine density
// from nvidia's toolkit
__device__ __inline__ void sampleUnitHemisphere( const optix::float2& sample,
                                                 const optix::float3& U,
                                                 const optix::float3& V,
                                                 const optix::float3& W,
                                                 optix::float3& point )
{
    using namespace optix;

    float phi = 2.0f * M_PIf * sample.x;
    float r = sqrt( sample.y );
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = 1.0f - x*x -y*y;
    z = z > 0.0f ? sqrt(z) : 0.0f;

    point = x*U + y*V + z*W;
}


__device__ void sampleHemisphere(float3& n, float3& newDir, unsigned int &seed) {

	float alpha, beta;

	alpha = rnd(seed) * 1.57 ;
	beta = rnd(seed) * 6.28;

	float x,y,z;
	x = sin(alpha) * sin(beta);
	z = sin(alpha) * cos(beta);
	y = cos(alpha);

	float3 U,V;
	createONB(n, U,V);
	newDir = y * n + x * V + z * U;
}


__device__ void sampleUnitHemisphereCosWeighted(float3& normal, float3& newDir, unsigned int& seed)
{
	float2 r;
	float3 U,V;

	r.x = rnd(seed);
	r.y = rnd(seed);
	//mapToDisk(r);
	createONB(normal, U, V);
	sampleUnitHemisphere(r, V, U, normal , newDir);
}


__device__ float4 sampleAreaLight(float3 surfaceNormal, float3 hitPoint, 
				 float3 lightP, float3 lightN, float lightSizeX,  float lightSizeY,unsigned int &seed)  
{
	PerRayDataResult shadow_prd;
    shadow_prd.result = make_float4(0.0f);
	float4 result = make_float4(0.0);
	unsigned int lightSamples = 1;
	//unsigned int seedi, seedj, a, b;
	float2 r;
	float dot1 = dot(hitPoint-lightP, lightN);
	// test if light is directed towards surface
	if (dot1 < 0)
		return result;
	
	for (unsigned int i = 0; i < lightSamples; ++i) 
	{
		//a = length(hitPoint) * 25676789;
		//b = i * 1027;
		//seedi = tea<16>(b,a);
		//seedj = tea<16>(b,a+1);
		r.x = rnd(seed);
		r.y = rnd(seed);
		//random2D(hitPoint, r);
		float3 lPos = lightP + make_float3(1,0,0) * lightSizeX * r.x + make_float3(0,0,1) * lightSizeY * r.y;
		float3 lDir = lPos - hitPoint;
		float lightDist = length(lDir);
		lDir = normalize(lDir);

		float NdotL = dot(surfaceNormal, lDir);
		if (NdotL > 0) {
			dot1 = max(0.0f,dot(lightN, -lDir));

			shadow_prd.result = make_float4(1.0f);
 			optix::Ray shadow_ray( hitPoint, lDir, Shadow, 0.1, lightDist+0.01 );
			rtTrace(top_object, shadow_ray, shadow_prd);
			result += shadow_prd.result * NdotL * dot1 ;
		}
	}
	result.w = lightSamples;
	//return(result);
	return(result/lightSamples);
}