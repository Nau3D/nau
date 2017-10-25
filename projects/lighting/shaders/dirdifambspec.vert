#version 330

uniform	mat4 m_pvm;
uniform	mat4 m_view;
uniform mat4 m_viewModel;
uniform	mat3 m_normal;

uniform	vec4 diffuse;
uniform	vec4 ambient;
uniform	vec4 specular;
uniform	float shininess;

uniform	vec4 l_dir;	   // global space

in vec4 position;	// local space
in vec3 normal;		// local space

// the data to be sent to the fragment shader
out Data {
	vec4 color;
} DataOut;



void main () {
	
	// set the specular term to black
	vec4 spec = vec4(0.0);

	vec3 n = normalize(m_normal * normal);
	vec3 ld = normalize(vec3(m_view * -l_dir));

	float intensity = max(dot(n, ld), 0.0);

	// if the vertex is lit compute the specular color
	if (intensity > 0.0) {
		// compute position in camera space
		vec3 pos = vec3(m_viewModel * position);
		// compute eye vector and normalize it 
		vec3 eye = normalize(-pos);
		// compute the half vector
		vec3 h = normalize(ld + eye);

		// compute the specular term into spec
		float intSpec = max(dot(h,n), 0.0);
		spec = specular * pow(intSpec, shininess);
	}
	// add the specular color when the vertex is lit
	DataOut.color = max(intensity *  diffuse + spec, diffuse * 0.25);

	// float len = length(DataOut.color.rgb);
	// float color;
	// if (len < 0.25)
		// color = 0.25;
	// else if (len < 0.5)
		// color = 0.5;
	// else if (len < 0.75)
		// color = 0.75;
	// else if (len < 0.9)
		// color = 0.9;
	// else color = 1;
	// DataOut.color = vec4(color);	
	gl_Position = m_pvm * position;	
}