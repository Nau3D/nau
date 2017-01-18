#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

struct PerRayDataResult
{
  float4 result;
};

rtDeclareVariable(float4, lightDir, , );

rtDeclareVariable(rtObject,      top_object, , );

rtBuffer<float4> vertex_buffer;     
rtBuffer<uint> index_buffer;

rtTextureSampler<float4,2> pos_buffer;

rtBuffer<float4,2> output0;


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(PerRayDataResult, prdr, rtPayload, );

rtDeclareVariable(int, Shadow, , );

RT_PROGRAM void buffer_camera(void)
{
	float4 i = tex2D( pos_buffer, launch_index.x, launch_index.y );
	PerRayDataResult prdr;	
	
	prdr.result = make_float4(1.0f);
	// w is greater than zero when pixel is facing the light
	if (i.w > 0.0f) {	
		float3 ray_origin = make_float3(i);
		float3 lDir = make_float3(-lightDir);
		optix::Ray ray = optix::make_Ray(ray_origin, lDir, 0, 0.0001f, RT_DEFAULT_MAX);
		rtTrace(top_object, ray, prdr);
	}
	else
		prdr.result = make_float4(1.0f);

	output0[launch_index] = prdr.result;

}


RT_PROGRAM void any_hit_shadow()
{
	prdr.result =  make_float4(0.0f);
	rtTerminateRay();
}




RT_PROGRAM void exception(void)
{
	output0[launch_index] = make_float4(1.f, 0.f, 0.f, 1.f);
}


RT_PROGRAM void geometryintersection(int primIdx)
{

	float4 vecauxa = vertex_buffer[index_buffer[primIdx*3]];
	float4 vecauxb = vertex_buffer[index_buffer[primIdx*3+1]];
	float4 vecauxc = vertex_buffer[index_buffer[primIdx*3+2]];

	float3 v0 = make_float3(vecauxa);
	float3 v1 = make_float3(vecauxb);
	float3 v2 = make_float3(vecauxc);

  // Intersect ray with triangle
  float3 n;
  float  t, beta, gamma;
  if( intersect_triangle( ray, v0, v1, v2, n, t, beta, gamma ) ) {

    if(  rtPotentialIntersection( t ) ) {

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
