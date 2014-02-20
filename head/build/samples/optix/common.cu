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

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );

rtDeclareVariable(rtObject,      top_object, , );

rtBuffer<float4> vertex_buffer;     
rtBuffer<float4> index_buffer;

rtBuffer<float4,2> output0;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );rtDeclareVariable(PerRayDataResult, prdr, rtPayload, );
rtDeclareVariable(rtObject, top_object, , );

RT_PROGRAM void pinhole_camera()
{
  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, 0.000000000000001, RT_DEFAULT_MAX);

  PerRayDataResult prd;

  rtTrace(top_object, ray, prd);

  output_buffer[launch_index] = make_color( prd.result );
}

RT_PROGRAM void shade()
{
	prdr.result = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
}


RT_PROGRAM void exception(void)
{
	output_buffer[launch_index] = make_float4(1.f, 0.f, 0.f, 1.f);
}

RT_PROGRAM void miss(void)
{
	prd_sr.shadowresult = make_float4(1.f);
}

RT_PROGRAM void geometryintersection(int primIdx)
{
	float4 vecauxa = vertex_buffer[primIdx*3];
	float4 vecauxb = vertex_buffer[primIdx*3+1];
	float4 vecauxc = vertex_buffer[primIdx*3+2];
	float3 e1, e2, h, s, q;
	float a,f,u,v,t;

	float3 v0 = make_float3(vecauxa);
	float3 v1 = make_float3(vecauxb);
	float3 v2 = make_float3(vecauxc);

	e1 = v1 - v0;
	e2 = v2 - v0;

	h = cross(ray.direction,e2);
	a = dot(e1,h);

	if (a <= -0.00001 || a >= 0.00001)
	{
		f = 1.0/a;
		s = ray.origin - v0;
		u = f * (dot(s,h));
	
		if (u >= 0.0 && u <= 1.0)
		{
			q = cross(s,e1);
			v = f * (dot(ray.direction,q));

			if (v >= 0.0 && u + v <= 1.0)
			{
				t = f * (dot(e2,q));

				if (t > 0.00001)
				{
					rtReportIntersection(0);
				}
			}
		}
	}
}

RT_PROGRAM void boundingbox(int primIdx, float result[6])
{
	float4 vecauxa = vertex_buffer[primIdx*3];
	float4 vecauxb = vertex_buffer[primIdx*3+1];
	float4 vecauxc = vertex_buffer[primIdx*3+2];
	
	float3 v0 = make_float3(vecauxa);
	float3 v1 = make_float3(vecauxb);
	float3 v2 = make_float3(vecauxc);
	
	optix::Aabb* aabb = (optix::Aabb*)result;
	aabb->set( v0, v1, v2 );
}

