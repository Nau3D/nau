#version 330

// the matrices
uniform	mat4 m_pvm;
uniform	mat4 m_viewModel;
uniform	mat4 m_view;
uniform	mat3 m_normal;

uniform	vec4 l_dir;	// global space

// vertex attributes
in vec4 position;	// local space
in vec3 normal;		// local space
in vec2 texCoord0;

// data to be sent to the fragment shader
out Data {
	vec3 eye;
	vec2 texCoord;
	vec3 l_dir;
	vec3 normal;
} DataOut;

void main () {
	// pass through texture coordinates
	DataOut.texCoord = texCoord0;
	
	// all vectors to camera space
	DataOut.normal = normalize(m_normal * normal);
	
	DataOut.eye = vec3(-(m_viewModel * position)); 
	DataOut.l_dir = normalize(vec3(m_view * -l_dir));

	gl_Position = m_pvm * position;	
}