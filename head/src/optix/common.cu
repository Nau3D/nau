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

rtDeclareVariable(rtObject,      top_object, , );

rtBuffer<float4> vertex_buffer;     
rtBuffer<uint> index_buffer;
rtBuffer<float4> normal;
rtBuffer<float4> texCoord0;

//rtTextureSampler<float4,2> pos_buffer;

rtTextureSampler<float4, 2> tex0;

rtBuffer<float4,2> output0;


struct PerRayDataResult
{
  float4 result;
	int depth;
};

struct PerRayData_shadow
{
  float4 result;
  float entrance;
};

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(PerRayDataResult, prdr, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(float3, texCoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

rtDeclareVariable(int, Phong, , );
rtDeclareVariable(int, Shadow, , );



#include "util.h"

RT_PROGRAM void pinhole_camera()
{
	float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U*fov + d.y*V*fov + W);
	
	optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, Phong, 0.00000000001, RT_DEFAULT_MAX);
	
	PerRayDataResult prd;
	prd.result = make_float4(1.0f);
	prd.depth = 0;
	
	rtTrace(top_object, ray, prd);

	output0[launch_index] = prd.result;
}

RT_PROGRAM void pinhole_camera_ms()
{
	float4 color;
	int sqrt_num_samples = 4;
	int samples = sqrt_num_samples * sqrt_num_samples;
	unsigned int seedi, seedj;

	float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;

	float2 scale = 1 / (make_float2(launch_dim) * sqrt_num_samples) * 2.0f;

	for (int i = 0; i < sqrt_num_samples; ++i) {
		for (int j = 0; j < sqrt_num_samples; ++j) {

			seedi = tea<16>(launch_dim.x*launch_index.y+launch_index.x,2*(i*sqrt_num_samples+j));
			seedj = tea<16>(launch_dim.x*launch_index.y+launch_index.x,2*(i*sqrt_num_samples+j)+1);
			float2 sample = d + make_float2((i+1)*rnd(seedi), (j+1)*rnd(seedj)) * scale;

			float3 ray_origin = eye;
			float3 ray_direction = normalize(sample.x*U*fov + sample.y*V*fov + W);
	
			optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, Phong, 0.000000001, RT_DEFAULT_MAX);
	
			PerRayDataResult prd;
			prd.result = make_float4(1.0f);
			prd.depth = 0;
	
			rtTrace(top_object, ray, prd);
			color += prd.result;
		}
	}
	output0[launch_index] = color / samples;
}


RT_PROGRAM void any_hit_shadow()
{
	prd_shadow.result =  make_float4(0.0f);
	rtTerminateRay();
}


RT_PROGRAM void keepGoingShadow() {

	float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float atenuation = 1.0;

	if (prd_shadow.entrance == 0.0) { // entrance
		atenuation *= sqrt(fabs(dot(n, ray.direction)));
		prd_shadow.entrance = t_hit;
	}
	else { // exit
		atenuation = pow(exp(log(0.84) * (fabs(t_hit - prd_shadow.entrance))),4);
		atenuation *= sqrt(fabs(dot(n, ray.direction)));
	}
	prd_shadow.result *= atenuation;

	rtIgnoreIntersection();
}


RT_PROGRAM void keepGoing() {

	prdr.result *= 0.9;
	rtIgnoreIntersection();
}


RT_PROGRAM void shade()
{
//	prdr.result = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
	PerRayData_shadow shadow_prd;
    shadow_prd.result = make_float4(1.0f);


	float3 lDir = make_float3(-lightDir);
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float NdotL = max(0.0f, dot(world_geometric_normal, lDir));
	if (NdotL > 0.0f) {

		float3 hit_point = ray.origin + t_hit * ray.direction;
		optix::Ray shadow_ray( hit_point, lDir, Shadow, 0.001, RT_DEFAULT_MAX );
		rtTrace(top_object, shadow_ray, shadow_prd);
	}
	float4 att = make_float4(1.0);
	att.x = shadow_prd.result.x * NdotL;
	att.y = shadow_prd.result.y * NdotL;
	att.z = shadow_prd.result.z * NdotL;
	NdotL *= shadow_prd.result.x;
	float4 color = diffuse* 1.3f;
	if (texCount > 0)
		color = color * tex2D( tex0, texCoord.x, texCoord.y );
	prdr.result *= make_float4(color.x*(0.3+att.x), color.y*(0.3+att.y), color.z*(0.3+att.z), 1.0f);
	//prdr.result *= make_float4(NdotL, NdotL, NdotL,1.0);
}


RT_PROGRAM void shadePointLight()
{
//	prdr.result = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
	PerRayData_shadow shadow_prd;
    shadow_prd.result = make_float4(1.0f);
	shadow_prd.entrance = 0.0f;

	float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 hit_point = ray.origin + t_hit * ray.direction;	

	float3 lDir = make_float3(lightPos) - hit_point;
	float lightDist = length(lDir);
	lDir = normalize(lDir);

	float NdotL = max(0.0f, dot(n, lDir));
	if (NdotL > 0.0f) {
		float3 hit_point = ray.origin + t_hit * ray.direction;
 
		optix::Ray shadow_ray( hit_point, lDir, Shadow, 0.000002, lightDist+0.01 );
		rtTrace(top_object, shadow_ray, shadow_prd);
	}
	float4 att = make_float4(1.0);
	att.x = shadow_prd.result.x * NdotL;
	att.y = shadow_prd.result.y * NdotL;
	att.z = shadow_prd.result.z * NdotL;
//	NdotL *= shadow_prd.result.x;
	float4 color = diffuse* 1.3f;
	if (texCount > 0)
		color = color * tex2D( tex0, texCoord.x, texCoord.y );
	prdr.result *= make_float4(color.x*(0.3+att.x), color.y*(0.3+att.y), color.z*(0.3+att.z), 1.0f);
	//prdr.result = make_float4(NdotL, NdotL, NdotL,1.0);
}


RT_PROGRAM void tracePathMetal() 
{
	if (prdr.depth < 4) {
	
		float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
		float3 hit_point = ray.origin + t_hit * ray.direction;
		float3 i = ray.direction;
		float3 r = optix::reflect(i, n);

		optix::Ray ray = optix::make_Ray(hit_point, r, Phong, 0.002, RT_DEFAULT_MAX);
	
		PerRayDataResult prd;
		prd.result = make_float4(1.0, 1.0, 1.0,1.0);
		prd.depth = prdr.depth+1;
		rtTrace(top_object, ray, prd);
		prdr.result = prd.result * dot(r,n);
	}
	else 
		prdr.result = make_float4(0.0);

//	prd.result = make_float4(1.0f);
//	prdr.result *= make_float4(0.0, 0.0, 1.0, 1.0);
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

		
		sampleUnitHemisphereCosLobe(r,exponent,hit_point,newDir);
		if (dot(newDir, n) <= 0.0)
			newDir = r;
		optix::Ray ray = optix::make_Ray(hit_point, newDir, Phong, 0.002, RT_DEFAULT_MAX);
	
		PerRayDataResult prd;
		prd.result = make_float4(1.0, 1.0, 1.0,1.0);
		prd.depth = prdr.depth+1;
		rtTrace(top_object, ray, prd);
		prdr.result = prd.result * (exponent+2) / (exponent+1) * dot(newDir,n);
	}
	else 
		prdr.result = make_float4(0.0);

//	prd.result = make_float4(1.0f);
//	prdr.result *= make_float4(0.0, 0.0, 1.0, 1.0);
} 


RT_PROGRAM void tracePath()
{
	float r;
	PerRayDataResult prdRec;
	prdRec.result = make_float4(0.0);
	float3 newDir;

	float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 hit_point = ray.origin + t_hit * ray.direction;	

	float4 shadow = sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0,-1,0), 0.4, 0.45); 

	random(hit_point,r);

	if (prdr.depth < 8 && r < 0.7f) {
//	if (prdr.depth < 8) {
		prdRec.depth = prdr.depth+1;
		prdRec.result = make_float4(1.0);
		sampleUnitHemisphereCosWeighted(n,hit_point,newDir);
		//float  seed = t_hit * 2789457;
		//sampleHemisphere(seed, n,newDir);
		optix::Ray newRay(hit_point, newDir, Phong, 0.00002, 5);
		rtTrace(top_object, newRay, prdRec);


		shadow += prdRec.result/0.7f;// * dot(newDir,n);
//		shadow += prdRec.result;// * dot(newDir,n);
	}
	float4 color = diffuse;
	if (texCount > 0)
		color = color * tex2D( tex0, texCoord.x, texCoord.y );
	prdr.result *= color * (shadow);
}


RT_PROGRAM void shadeAreaLight()
{
	float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 hit_point = ray.origin + t_hit * ray.direction;	

	float4 shadow = sampleAreaLight(n, hit_point, make_float3(lightPos), make_float3(0,-1,0), 0.4, 0.45); 

	float4 color = diffuse* 1.3f;
	if (texCount > 0)
		color = color * tex2D( tex0, texCoord.x, texCoord.y );
	prdr.result *= color * (0.3 + shadow);
}


RT_PROGRAM void shadeLight()
{
	prdr.result = make_float4(1.0);
}


RT_PROGRAM void shadeMetal() 
{

	float3 n = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 hit_point = ray.origin + t_hit * ray.direction;

	float3 i = ray.direction;
	float3 r = optix::reflect(i, n);

	optix::Ray ray = optix::make_Ray(hit_point, r, Phong, 0.002, RT_DEFAULT_MAX);
	
	PerRayDataResult prd;
	prd.result = make_float4(1.0, 1.0, 1.0,1.0);
	prd.depth = prdr.depth++;
	rtTrace(top_object, ray, prd);
	prdr.result = prd.result;

//	prd.result = make_float4(1.0f);
//	prdr.result *= make_float4(0.0, 0.0, 1.0, 1.0);
}


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

	optix::Ray ray = optix::make_Ray(hit_point, t, Phong, 0.002, RT_DEFAULT_MAX);
	
	PerRayDataResult prd;
	prd.result = make_float4(1.0, 1.0, 1.0,1.0);
	prd.depth = prdr.depth++;
	rtTrace(top_object, ray, prd);
	prdr.result = prd.result * atenuation;

//	prd.result = make_float4(1.0f);
//	prdr.result *= make_float4(0.0, 0.0, 1.0, 1.0);
}


RT_PROGRAM void shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.result = make_float4(0.1);

  rtTerminateRay();
}


RT_PROGRAM void exception(void)
{
	output0[launch_index] = make_float4(1.f, 0.f, 0.f, 1.f);
}


RT_PROGRAM void miss(void)
{
	prdr.result = make_float4(0.0f);
}


RT_PROGRAM void missShadow(void)
{
	prdr.result = make_float4(1.0f);
}


// Alpha test program
RT_PROGRAM void alpha_test()
{
	if (tex2D( tex0, texCoord.x, texCoord.y ).w < 0.25f)
		rtIgnoreIntersection();
}


// Alpha test shadow
RT_PROGRAM void alpha_test_shadow()
{
	if (tex2D( tex0, texCoord.x, texCoord.y ).w < 0.25f)
		rtIgnoreIntersection();
	else {
		prd_shadow.result = make_float4(0.0);
		rtTerminateRay();
	}

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


//RT_PROGRAM void boundingbox(int primIdx, float result[6])
//{
//	float3 v0 = make_float3(vertex_buffer[index_buffer[primIdx*3]]);
//	float3 v1 = make_float3(vertex_buffer[index_buffer[primIdx*3+1]]);
//	float3 v2 = make_float3(vertex_buffer[index_buffer[primIdx*3+2]]);
//	
//	optix::Aabb* aabb = (optix::Aabb*)result;
//	aabb->set( v0, v1, v2 );
//}

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
