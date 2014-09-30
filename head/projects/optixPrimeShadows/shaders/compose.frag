#version 430

struct hit {
	float t;
	int t_id;
	float u,v;
};

layout(std430, binding = 2) buffer hitsBuffer {
	hit hits[];
//	vec4 hits[];
};

uniform sampler2D texColor;
uniform sampler2D texNormal;
uniform vec3 lightDirection;

in vec2 texCoordV;
out vec4 colorOut;

void main() {
	ivec2 coord = ivec2(texCoordV * ivec2(1024,1024));
	int coordB = coord.x* 1024 + coord.y;

	 hit h = hits[coordB];
//	 vec4 h = hits[coordB];
	
	vec4 color = texture(texColor, texCoordV);
	vec3 n = texture(texNormal, texCoordV).xyz;

	float intensity = max(0.0, dot(n, -normalize(lightDirection)));
	color = vec4(color.rgb*intensity+0.2, color.a);
	
	 if (h.t_id >= 0)
//	if (h.x > 0)
		colorOut = color * vec4(0.5, 0.5, 0.5, 1.0);
	 else
		 colorOut = color * vec4(1.0, 1.0, 1.0, 1.0);
}	