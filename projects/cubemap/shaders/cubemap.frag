#version 420

uniform samplerCube texUnit; 

in vec3 normalV;
in vec3 eyeV;

out vec4 outColor;
//out vec4 color2;

void main() {

	vec3 n = normalize(normalV);
	vec3 e = normalize(eyeV);
	vec3 t = reflect(e, n);
	vec3 ref = texture(texUnit, t).rgb;

	float eta = 1.1;
	vec3 r = refract(e, n, eta);
	vec3 g = refract(e, n, eta+0.02);
	vec3 b = refract(e, n, eta+0.04);
	float k = 1.0 - eta * eta * (1.0 - dot(n, e) * dot(n, e));
	vec3 refract;
//	if (r * g * b != vec3(0.0))

		refract = vec3(texture(texUnit, r).r, texture(texUnit, g).g, texture(texUnit, b).b);
//	else
//		refract = vec3(0.0);

	float fresnel = dot(-normalize(eyeV), normalize(normalV));

	//color1 = vec4(k * refract + (1-k) * reflect, 1.0) ;
	//color2 = vec4(vec3(k), 1.0);//vec4(refract, 1.0);
	if (textureSize(texUnit,0).x == 1)
		outColor = vec4(0.5);
	else
		outColor = vec4(ref, 1.0);
	
	//outColor = vec4(texture(texUnit, vec3(0.5,0.5,0.5)));
}