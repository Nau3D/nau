#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb.h>
#include <optixu/optixu_aabb_namespace.h>
#include "random.h"


using namespace optix;

// Interface variables

// Camera
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         fov, , );

// Material
rtDeclareVariable(float4, diffuse, , );
rtDeclareVariable(int, texCount, , );


// Light
rtDeclareVariable(float4, lightDir, , );
rtDeclareVariable(float4, lightPos, , );
rtDeclareVariable(uint, frameCount, , );
rtDeclareVariable(int, trace, , );

rtDeclareVariable(rtObject,      top_object, , );

rtBuffer<float4> vertex_buffer;     
rtBuffer<uint> index_buffer;
rtBuffer<float4> normal;
rtBuffer<float4> texCoord0;

rtTextureSampler<float4, 2> tex0;

rtBuffer<float4,2> output0;

struct PerRayDataResult
{
	float4 result;
	int depth;
	unsigned int seed;
	float entrance;
  };


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(PerRayDataResult, prdr, rtPayload, );
//rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(float3, texCoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

rtDeclareVariable(int, Phong, , );
rtDeclareVariable(int, Shadow, , );

#include "util.h"

RT_PROGRAM void pinhole_camera_ms()
{
	float4 color = make_float4(0.0);
	int sqrt_num_samples = 2;
	int samples = sqrt_num_samples * sqrt_num_samples;
	unsigned int seedi, seedj;

	size_t2 screen = output0.size();

	float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
	float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

	float2 jitter_scale = inv_screen / sqrt_num_samples;
	float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;

	float2 scale = 1 / (make_float2(launch_dim) * sqrt_num_samples) * 2.0f;
	unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frameCount);

	for (int i = 0; i < sqrt_num_samples; ++i) {
		for (int j = 0; j < sqrt_num_samples; ++j) {

			float2 jitter = make_float2((i+1)+rnd(seed), (j+1)+rnd(seed));
			float2 d = pixel + jitter*jitter_scale;
			float3 ray_origin = eye;
			float3 ray_direction = normalize(d.x*U*fov + d.y*V*fov + W);
			
			optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, Phong, 0.000000001, RT_DEFAULT_MAX);
	
			PerRayDataResult prd;
			prd.result = make_float4(1.0f);
			prd.depth = 0;
			prd.seed = seed;
	
			rtTrace(top_object, ray, prd);
			color += prd.result;
		}
	}
	output0[launch_index] = color/samples;
}


RT_PROGRAM void any_hit_shadow()
{
	prdr.result =  make_float4(0.0f);
	rtTerminateRay();
}


RT_PROGRAM void missShadow(void)
{
	prdr.result = make_float4(1.0f);
}


RT_PROGRAM void keepGoingShadow() 
{
	float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float atenuation = 1.0;

	if (prdr.entrance == 0.0) { // entrance
		atenuation *= sqrt(fabs(dot(n, ray.direction)));
		prdr.entrance = t_hit;
	}
	else { // exit
		atenuation = pow(exp(log(0.84) * (fabs(t_hit - prdr.entrance))),4);
		atenuation *= sqrt(fabs(dot(n, ray.direction)));
	}
	prdr.result *= atenuation;

	rtIgnoreIntersection();
}

RT_PROGRAM void tracePathMetal() 
{
	if (prdr.depth < 4) {
	
		float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
		float3 hit_point = ray.origin + t_hit * ray.direction;
		float3 i = ray.direction;
		float3 r = optix::reflect(i, n);

		optix::Ray ray = optix::make_Ray(hit_point, r, Phong, 0.000002, RT_DEFAULT_MAX);
	
		PerRayDataResult prd;
		prd.result = make_float4(1.0, 1.0, 1.0,1.0);
		prd.depth = prdr.depth+1;
		prd.seed = prdr.seed;
		rtTrace(top_object, ray, prd);
		prdr.result = prd.result;// * dot(r,n);
	}
	else 
		prdr.result = make_float4(0.0);
}


RT_PROGRAM void tracePathGlossy() 
{
	if (prdr.depth < 4) {
	
		float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
		float3 hit_point = ray.origin + t_hit * ray.direction;
		float3 i = ray.direction;
		float3 r = optix::reflect(i, n);
		
		float3 newDir;
		float exponent = 100;
		sampleUnitHemisphereCosLobe(r,exponent, newDir, prdr.seed);

		if (dot(newDir, n) <= 0.0)
			newDir = r;

		optix::Ray ray = optix::make_Ray(hit_point, newDir, Phong, 0.000002, RT_DEFAULT_MAX);
	
		PerRayDataResult prd;
		prd.result = make_float4(1.0, 1.0, 1.0,1.0);
		prd.depth = prdr.depth+1;
		prd.seed = prdr.seed;
		rtTrace(top_object, ray, prd);
		prdr.result = prd.result ;
	}
	else 
		prdr.result = make_float4(0.0);

} 


RT_PROGRAM void tracePath()
{
	PerRayDataResult prdRec;
	prdRec.result = make_float4(0.0);
	float3 newDir;

	float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 hit_point = ray.origin + t_hit * ray.direction;	

	float4 shadow = sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0,-1,0), 0.4, 0.45, prdr.seed); 

	float4 color = diffuse;
	
	float p = max(diffuse.x, max(diffuse.y, diffuse.z));
	if (prdr.depth > 1 || rnd(prdr.seed) > p)
		color = color * 1.0/(1-p);
	else {
	//if (prdr.depth < 2 ) {//&& r < 0.7f) {
//	if (prdr.depth < 8) {
		prdRec.depth = prdr.depth+1;
		prdRec.result = make_float4(1.0);
		prdRec.seed = prdr.seed;
		sampleUnitHemisphereCosWeighted(n,newDir,prdr.seed);
		//sampleHemisphere(n,newDir,prdr.seed);
		//float  seed = t_hit * 2789457;
		//sampleHemisphere(seed, n,newDir);
		optix::Ray newRay(hit_point, newDir, Phong, 0.2, 5000000);
		rtTrace(top_object, newRay, prdRec);


		shadow += prdRec.result;// * dot(newDir,n);
//		shadow += prdRec.result;// * dot(newDir,n);
	}
	if (texCount > 0)
		color = color * tex2D( tex0, texCoord.x, texCoord.y );
	prdr.result *= color * shadow;
	//prdr.result = color;
}


RT_PROGRAM void shadeLight()
{
	prdr.result = make_float4(1.0);
}


// RT_PROGRAM void shadeAreaLight()
// {
	// float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	// float3 hit_point = ray.origin + t_hit * ray.direction;	

	// float4 shadow = sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0,-1,0), 0.4, 0.45, prdr.seed); 

	// float4 color = diffuse* 1.3f;
	// if (texCount > 0)
		// color = color * tex2D( tex0, texCoord.x, texCoord.y );
	// prdr.result *= color * (0.3 + shadow);
// }

RT_PROGRAM void shadeGlass() 
{
	float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 hit_point = ray.origin + t_hit * ray.direction;
	float3 i = ray.direction;
	float3 t;
	float atenuation = 1.0;

	if (dot(n,i) < 0) // entrance ray
	{
		optix::refract(t, i, n, 1.5);
	}
	else // exiting ray
	{
		optix::refract(t, i, -n, 0.66);
		atenuation = exp(log(0.84) * t_hit);
	}

	optix::Ray ray = optix::make_Ray(hit_point, t, Phong, 0.000002, RT_DEFAULT_MAX);
	
	PerRayDataResult prd;
	prd.result = make_float4(1.0, 1.0, 1.0,1.0);
	prd.depth = prdr.depth;
	rtTrace(top_object, ray, prd);
	prdr.result = prd.result * atenuation;

//	prd.result = make_float4(1.0f);
//	prdr.result *= make_float4(0.0, 0.0, 1.0, 1.0);
}

RT_PROGRAM void exception(void)
{
	output0[launch_index] = make_float4(1.f, 0.f, 0.f, 1.f);
}


RT_PROGRAM void miss(void)
{
	prdr.result = make_float4(0.0, 0.0, 0.0, 0.0);
}

RT_PROGRAM void geometryintersection(int primIdx)
{

	float4 vecauxa = vertex_buffer[index_buffer[primIdx*3]];
	float4 vecauxb = vertex_buffer[index_buffer[primIdx*3+1]];
	float4 vecauxc = vertex_buffer[index_buffer[primIdx*3+2]];
//	float3 e1, e2, h, s, q;
//	float a,f,u,v,t;

	float3 v0 = make_float3(vecauxa);
	float3 v1 = make_float3(vecauxb);
	float3 v2 = make_float3(vecauxc);

  // Intersect ray with triangle
  float3 n;
  float  t, beta, gamma;
  if( intersect_triangle( ray, v0, v1, v2, n, t, beta, gamma ) ) {

    if(  rtPotentialIntersection( t ) ) {

      float3 n0 = make_float3(normal[ index_buffer[primIdx*3]]);
      float3 n1 = make_float3(normal[ index_buffer[primIdx*3+1]]);
      float3 n2 = make_float3(normal[ index_buffer[primIdx*3+2]]);

	  float3 t0 = make_float3(texCoord0[ index_buffer[primIdx*3]]);
	  float3 t1 = make_float3(texCoord0[ index_buffer[primIdx*3+1]]);
	  float3 t2 = make_float3(texCoord0[ index_buffer[primIdx*3+2]]);

      shading_normal   = normalize( n0*(1.0f-beta-gamma) + n1*beta + n2*gamma );
	  texCoord =  t0*(1.0f-beta-gamma) + t1*beta + t2*gamma ;
      geometric_normal = normalize( n );

	  rtReportIntersection(0);
    }
  }
}

RT_PROGRAM void boundingbox(int primIdx, float result[6])
{
	float3 v0 = make_float3(vertex_buffer[index_buffer[primIdx*3]]);
	float3 v1 = make_float3(vertex_buffer[index_buffer[primIdx*3+1]]);
	float3 v2 = make_float3(vertex_buffer[index_buffer[primIdx*3+2]]);  
	
	const float  area = length(cross(v1-v0, v2-v0));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if(area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf( fminf( v0, v1), v2 );
		aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
	} 
	else {
	    aabb->invalidate();
	}
}




