#version 430

struct hit {
	float t;
	int t_id;
	float u,v;
};

layout(std430, binding = 2) buffer hitsBuffer {
	hit hits[];
};

uniform sampler2D texColor;
uniform sampler2D texNormal;

uniform vec3 lightDir;
uniform int size;
uniform mat4 V;

in vec2 texCoordV;
out vec4 colorOut;

void main() {

	hit h;
	ivec2 coord = ivec2(texCoordV * ivec2(size,size));
	int coordB = coord.x * size + coord.y;

	h = hits[coordB];
	
	vec4 color = texture(texColor, texCoordV);
	vec3 n = texture(texNormal, texCoordV).xyz * 2.0 - 1.0;

	vec3 ld = vec3(V * vec4(-lightDir, 0.0));
	float intensity = max(0.0, dot(n, normalize(ld)));
	color = vec4(color.rgb*intensity+0.2, color.a);
	
	 if (h.t_id >= 0)
		colorOut = color * vec4(0.5, 0.5, 0.5, 1.0);
	 else
		 colorOut = color * vec4(1.0, 1.0, 1.0, 1.0);
		 
	//colorOut = vec4(intensity);	 
	 
}	