#version 150

uniform	mat4 m_pvm;
uniform mat4 m_view;
uniform	mat3 m_normal;
uniform vec4 l_dir; // global space

in vec4 position;	// local space
in vec3 normal;		// local space

// the data to be sent to the fragment shader
out Data {
	vec3 normal;
	vec3 l_dir;
} DataOut;

void main () {
	
	DataOut.normal = normalize(m_normal * normal);
	DataOut.l_dir = normalize(vec3(m_view * -l_dir));

	gl_Position = m_pvm * position;	
}