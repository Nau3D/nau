#version 440

out vec4 FragColor;

//layout (rgba8) uniform coherent volatile image3D grid;
layout (r32ui) uniform coherent volatile uimage3D grid;

uniform mat4 VM;
uniform vec2 WindowSize;
uniform vec4 RayDirection, CamPos;
uniform int GridSize;
uniform float top, bottom,left, right,near;
uniform vec4 rightV, upV;

uniform int level = 0;
uniform sampler2D texRSM, texPos;

in vec2 texCoordV;

struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

vec4 convRGBA8ToVec4(uint val)
{
	return vec4(float((val & 0x000000FF)), float((val & 0x0000FF00) >> 8U), float((val & 0x00FF0000) >> 16U), float((val & 0xFF000000) >> 24U));
}

uint convVec4ToRGBA8(vec4 val)
{
	return (uint(val.w) & 0x000000FF) << 24U | (uint(val.z) & 0x000000FF) << 16U | (uint(val.y) & 0x000000FF) << 8U | (uint(val.x) & 0x000000FF);
}

void imageAtomicRGBA8Avg(ivec3 coords, vec4 value)
{
	value.rgb *= 255.0;
	uint newVal = convVec4ToRGBA8(value);
	uint prevStoredVal = 0;
	uint curStoredVal;

	while((curStoredVal = imageAtomicCompSwap(grid, coords, prevStoredVal, newVal)) != prevStoredVal)
	{
		prevStoredVal = curStoredVal;
		vec4 rval = convRGBA8ToVec4(curStoredVal);
		rval.rgb = (rval.rgb * rval.a);	// Denormalize
		vec4 curValF = rval + value;	// Add
		curValF.rgb /= curValF.a;	// Renormalize
		newVal = convVec4ToRGBA8(curValF);
	}
}

bool IntersectBox(Ray r, AABB aabb, out float t0, out float t1)
{
    vec3 invR = 1.0 / r.Dir;
    vec3 tbot = invR * (aabb.Min-r.Origin);
    vec3 ttop = invR * (aabb.Max-r.Origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    t1 = min(t.x, t.y);
    return t0 <= t1;
}

void main()
{
	vec4 pos = texture(texPos, texCoordV);
	vec4 color = texture(texRSM, texCoordV);
	//color.xyz *= color.w;
	ivec3 coord = ivec3((pos*0.5 + 0.5)*GridSize);

	if (color.xyz != vec3(0,0,0))
		imageAtomicRGBA8Avg(coord, vec4(color.xyz,1));
		
	//FragColor = vec4(1,0,0,0);	
	//return;
/*		
	int numSamples = GridSize*2;//64;//int(WindowSize.x);
	
	float stepSize = 1/float(numSamples);
	vec4 dirV = upV * (top-bottom);
	vec4 dirH = rightV * (right-left);
	vec4 corner = CamPos + near*RayDirection - dirV*0.5 - dirH*0.5;
	vec4 rayOrigin = corner + dirV * (texCoordV.t) + dirH * (texCoordV.s);
    vec4 rayDirection = RayDirection;
//    rayDirection = (vec4(rayDirection, 0) * VM).xyz;

    Ray eye = Ray( rayOrigin.xyz, normalize(rayDirection.xyz) );
    AABB aabb = AABB(vec3(-1.0), vec3(+1.0));

    float tnear, tfar;
    IntersectBox(eye, aabb, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;
	
    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    rayStart = 0.5 * (rayStart + 1.0);
    rayStop = 0.5 * (rayStop + 1.0);

    vec3 pos = rayStart;
    vec3 step = normalize(rayStop-rayStart) * stepSize;
    float travel = distance(rayStop, rayStart);
	vec4 density = vec4(0);
    for (;  density.w == 0  &&  travel > 0.0;  travel -= stepSize) {

        //density = texture(grid, pos*0.5 + 0.5) ;
		density = convRGBA8ToVec4(imageLoad(grid, ivec3(pos*GridSize) ).x) ;
		//density = imageLoad(grid, ivec3(pos*GridSize) );
		pos += step;
		//density=vec4(1,0,0,0);
    }
	//memoryBarrier();
	//if (density.w > 0) 
	{
		vec4 color = texture(texRSM, texCoordV);
		imageAtomicRGBA8Avg(ivec3((pos-step)*GridSize), vec4(color.xyz,1));
		//imageStore(grid, ivec3((pos-step)*GridSize), vec4(color.xyz,1));
		//imageStore(grid, ivec3(10,10,10), vec4(1,0,0,1));
		FragColor = vec4(color);;
	}
*/
}