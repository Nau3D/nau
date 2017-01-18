#version 330

in vec3 lDirV; // camera space
in vec3 normalV; // camera space
in vec3 eyeV;

uniform float beta = 0.6;
uniform float alpha = 0.4;
uniform float b = 0.4;
uniform float y = 0.4;

uniform vec4 kCool = vec4(0.0, 0.0, 1.0, 1.0);
uniform vec4 kWarm = vec4(1.0, 1.0, 0.0, 1.0);


out vec4 colorOut;

void main() {

	// set the specular term to black
	vec4 spec = vec4(0.0);

	// normalize both input vectors
	vec3 n = normalize(normalV);
	vec3 e = normalize(vec3(eyeV));

	// intensity goes from -1 to 1
	float intensity = dot(n,lDirV);
	if (intensity > 0) {
		// compute the half vector
		vec3 h = normalize(lDirV + e);	
		// compute the specular term into spec
		float intSpec = max(dot(h,n), 0.0);
		spec = vec4(1.0) * pow(intSpec,128);
	}
	float blendValue = (1 + intensity) * 0.5;
	vec4 goochColor = mix(kCool * b + alpha, kWarm * y + beta, blendValue);

	colorOut = goochColor + spec;
}