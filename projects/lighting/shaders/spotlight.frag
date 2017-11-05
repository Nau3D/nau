#version 330

out vec4 colorOut;

layout (std140) uniform Material {
	vec4 diffuse;
	vec4 ambient;
	vec4 specular;
	vec4 emissive;
	float shininess;
	int texCount;
} Mat;

layout (std140) uniform LightSpot {
	vec4 l_pos, l_spotDir;
	float l_spotCutOff, l_spotExponent;
};

in Data {
	vec3 normal;
	vec3 eye;
	vec3 lightDir;
	vec3 spotDir;
} DataIn;


void main() {

	float intensity = 0.0;
	vec4 spec = vec4(0.0);
	float spotAttenuation = 0.0;

	vec3 ld = normalize(DataIn.lightDir);
	vec3 sd = normalize(DataIn.spotDir);
	
	// inside the cone?
	if (dot(sd,-ld) > l_spotCutOff) {
		
		vec3 n = normalize(DataIn.normal);
		intensity = max(dot(n,ld), 0.0);
		spotAttenuation = pow(dot(sd,-ld), l_spotExponent);

		
		if (intensity > 0.0) {
			vec3 eye = normalize(DataIn.eye);
			vec3 h = normalize(ld + eye);
			float intSpec = max(dot(h,n), 0.0);
			spec =  Mat.specular * pow(intSpec, Mat.shininess);
		}
	}
	
	
	colorOut = max(spotAttenuation * (intensity * Mat.diffuse + spec), Mat.diffuse * 0.25);

}