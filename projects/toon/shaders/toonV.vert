#version 330

uniform mat4 m_pvm;
uniform	mat4 m_view;
uniform	mat3 m_normal;

uniform	vec4 diffuse;
uniform	vec4 ambient;

uniform	vec4 l_dir;	   // camera space

in vec4 position;	// local space
in vec3 normal;		// local space

// the data to be sent to the fragment shader
out Data {
	vec4 color;
} DataOut;

void main () {
	// transform normal to camera space and normalize it
	vec3 n = normalize(m_normal * normal);
	vec3 ld = normalize(vec3(m_view * -l_dir));

	// compute the intensity as the dot product
	// the max prevents negative intensity values
	float intensity = max(dot(n, ld), 0.0);

	// compute the color 
	if (intensity > 0.9)
		DataOut.color = diffuse;
	else if (intensity > 0.75)
		DataOut.color = 0.75 * diffuse;
	else if (intensity > 0.5)
		DataOut.color = 0.5 * diffuse;
	else if (intensity > 0.25)
		DataOut.color = 0.25 * diffuse;
	else
		DataOut.color = 0.1 * diffuse;
		

	// transform the vertex coordinates
	gl_Position = m_pvm * position;	
}