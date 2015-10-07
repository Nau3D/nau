#version 330

layout (std140) uniform Matrices {
	mat4 m_pvm;
	mat4 m_viewModel;
	mat3 m_normal;
};

layout (std140) uniform Lights {
	vec4 l_pos, l_spotDir;
	float l_spotCutOff;
};

in vec4 position;
in vec3 normal;

out Data {
	vec3 normal;
	vec2 texCoord;
	vec3 eye;
	vec3 lightDir;
} DataOut;

void main () {

	vec4 pos = m_viewModel * position;

	DataOut.normal = normalize(m_normal * normal);
	DataOut.lightDir = vec3(l_pos-pos);
	DataOut.eye = vec3(-pos);


	gl_Position = m_pvm * position;	
}