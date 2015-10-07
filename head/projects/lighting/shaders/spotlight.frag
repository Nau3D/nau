#version 330

out vec4 colorOut;

layout (std140) uniform Material {
	vec4 diffuse;
	vec4 ambient;
	vec4 specular;
	vec4 emissive;
	float shininess;
	int texCount;
};

layout (std140) uniform Lights {
	vec4 l_pos, l_spotDir;
	float l_spotCutOff;
};

in Data {
	vec3 normal;
	vec2 texCoord;
	vec3 eye;
	vec3 lightDir;
} DataIn;


void main() {

	float intensity = 0.0;
	vec4 spec = vec4(0.0);

	vec3 ld = normalize(DataIn.lightDir);
	vec3 sd = normalize(vec3(-l_spotDir));	
	if (dot(sd,ld) > l_spotCutOff) {
		
	vec3 n = normalize(DataIn.normal);
	// inside the cone?

		intensity = max(dot(n,ld), 0.0);
		
		if (intensity > 0.0) {
			vec3 eye = normalize(DataIn.eye);
			vec3 h = normalize(ld + eye);
			float intSpec = max(dot(h,n), 0.0);
			spec = specular * pow(intSpec, shininess);
		}
	}
	
	colorOut = max(intensity * diffuse + spec, ambient);
}