#version 330

layout (std140) uniform Material {
	vec4 diffuse;
	vec4 ambient;
	vec4 specular;
	vec4 emissive;
	float shininess;
};

in Data {
	vec3 normal;
	vec3 eye;
	vec3 lightDir;
} DataIn;

out vec4 colorOut;

void main() {

	vec4 spec = vec4(0.0);

	vec3 n = normalize(DataIn.normal);
	vec3 l = normalize(DataIn.lightDir);
	vec3 e = normalize(DataIn.eye);

	// distance to the light
	float dist = length(DataIn.lightDir);

	float intensity = max(dot(n,l), 0.0) ;
	
	if (intensity > 0.0) {

		vec3 h = normalize(l + e);
		float intSpec = max(dot(h,n), 0.0);
		spec = specular * pow(intSpec, shininess);
	}
	// attenuation
	float attenuation = 1.0;///(0.5 * dist*dist);
	
	colorOut = max((intensity * diffuse + spec) * attenuation , diffuse * 0.25);
	//colorOut = vec4(intensity);
}