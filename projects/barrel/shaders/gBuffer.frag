#version 330

in Data {
	vec4 eye;
	vec2 texCoord;
	vec3 normal;
	vec3 tangent;
	vec3 bitangent;
	vec3 l_dir;
	vec3 pos;
	flat int tex;
} DataIn;

layout (location = 0) out vec4 normal;
layout (location = 1) out vec4 texCoord;
layout (location = 2) out vec4 tangent;
layout (location = 3) out vec4 pos;
layout (location = 4) out vec4 eye;


void main() {

	// normalize vectors
	
	normal = vec4(normalize(DataIn.normal) * 0.5 + 0.5, 0);
	tangent = vec4(normalize(DataIn.tangent) * 0.5 + 0.5, 0);
	
	eye = DataIn.eye;
	texCoord = vec4(DataIn.texCoord, 0, 0);
	pos = vec4(DataIn.pos, DataIn.tex);
}
