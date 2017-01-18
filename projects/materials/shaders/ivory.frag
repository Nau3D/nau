#version 330

in vec3 lDirV; // camera space
in vec3 normalV;
in vec3 eyeV;

out vec4 colorOut;

void main() {

	// set the specular term to black
	float spec = 0.0;

	// normalize both input vectors
	vec3 n = normalize(normalV);
	vec3 e = normalize(vec3(eyeV));

	float intensity = max(dot(n,lDirV), 0.0);

	// "half-lambert" technique
	intensity = intensity * 0.5 + 0.5;

	// if the vertex is lit compute the specular color
	if (intensity > 0.0) {
		// compute the half vector
		vec3 h = normalize(lDirV + e);	
		// compute the specular term into spec
		float intSpec = max(dot(h,n), 0.0);
		spec = pow(intSpec,64);
	}
	colorOut = vec4(intensity + spec);
}