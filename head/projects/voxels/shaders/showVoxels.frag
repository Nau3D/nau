#version 440

out vec4 FragColor;

uniform sampler3D grid;
uniform mat4 VM;
uniform float FOV;
uniform vec2 WindowSize;
uniform vec3 RayOrigin;
uniform int GridSize;
uniform int level = 3;

struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

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
	
	int numSamples = GridSize;//64;//int(WindowSize.x);
	float stepSize = 1/float(numSamples);
	float FocalLength = 1.0/ tan(radians(FOV*0.5));
    vec3 rayDirection;
    rayDirection.xy = 2.0 * gl_FragCoord.xy / WindowSize - 1.0;
    rayDirection.z = -FocalLength;
    rayDirection = (vec4(rayDirection, 0) * VM).xyz;

    Ray eye = Ray( RayOrigin, normalize(rayDirection) );
    AABB aabb = AABB(vec3(-1.0), vec3(+1.0));

    float tnear, tfar;
    IntersectBox(eye, aabb, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;
	
    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    //rayStart = 0.5 * (rayStart + 1.0);
    //rayStop = 0.5 * (rayStop + 1.0);

    vec3 pos = rayStart;
    vec3 step = normalize(rayStop-rayStart) * stepSize;
    float travel = distance(rayStop, rayStart);
	vec4 density = vec4(0);
    for (;  density.w == 0  &&  travel > 0.0;  travel -= stepSize) {

        //density = texture(grid, pos*0.5 + 0.5) ;
		density = texelFetch(grid, ivec3((pos*0.5 + 0.5) * GridSize/pow(2.0,level)), level) ;
		pos += step;
     }

    FragColor.rgb = vec3(density)/density.w;
    FragColor.a = 1;
	
	//FragColor = vec4(1.0);
}