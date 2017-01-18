#version 420 

uniform vec4 lightDirection, lightColor;
uniform vec4 diffuse, ambient, emission, specular;
uniform float shininess;

in vec3 Normal;
in vec3 LightDirection;
in vec3 Eye;
out vec4 outColor;

void main()
{
	vec4 spec = vec4(0.0);
	vec3 n = normalize(Normal);
	vec3 l = -normalize(LightDirection);
	vec3 e = normalize(Eye);
	float intensity = max(dot(n,l), 0.0);
	if (intensity > 0.0) {
		vec3 h = normalize(l + e);
		float intSpec = max(dot(h,n), 0.0);
		spec = specular * pow(intSpec, shininess);
	}
	vec4 aux = max(intensity * diffuse + spec, ambient);
	//vec4 temp = vec4(0.0, 1.0, 0.0, 0.4);
	//outColor = temp;
	outColor = vec4(aux.xyz, 0.3);
	//outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
