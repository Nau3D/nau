#version 430

uniform vec3 lightDirection;
uniform sampler2D texUnit;
uniform int size;

struct ray {
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
	r.dir = vec4(-lightDirection, 1000.0);
	
	ivec2 coord = ivec2(texCoordV*vec2(size,size));
	int coordB = coord.x* size + coord.y;
	rays[coordB] = r;

	 outColor=r.pos;
}