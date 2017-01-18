#version 150

uniform	mat4 m_pvm;
uniform mat4 m_view;
uniform	mat3 m_normal;
uniform vec4 l_dir; // global space

in vec4 position;
in vec3 normal;

out float intensityV;

void main() {
	// transform and normalise both vectors
	vec3 n = normalize(m_normal * normal);

// compute the intensity using the dot operation
	intensityV = dot(n, normalize(vec3(m_view * -l_dir)));

	gl_Position = m_pvm * position;
}