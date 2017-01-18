#version 430

uniform vec3 lightDirection;
uniform sampler2D texUnit;
uniform int RenderTargetX;
uniform int RenderTargetY;
uniform float tMax;

struct ray {
	// vec3 pos;
	// float tmin;
	// vec3 dir;
	// float tmax;
	vec4 pos;
	vec4 dir;
};

layout(std430, binding = 1) buffer RayBuffer {
	ray rays[];
};

in vec2 texCoordV;

out vec4 outColor;

void main()
{
	
	vec3 pos = texture(texUnit, texCoordV).xyz;
	
	ray r;
	r.pos = vec4(pos, 0.01);
	r.dir = vec4(-lightDirection, tMax);
	
	//(x*height + y) or (x+y*width)
	ivec2 coord = ivec2(texCoordV.x*RenderTargetX,texCoordV.y*RenderTargetY);
	int coordB = coord.x * RenderTargetY + coord.y; 
	//int coordB = coord.x + coord.y * RenderTargetX;
	rays[coordB] = r;

	outColor=r.pos;
}