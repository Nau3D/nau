// http://www.renderman.org/RMR/Shaders/SHWShaders/SHW_brushedmetal.sl


#version 330

uniform vec4 diffuse = vec4(0.8, 0.8, 0.8, 1.0);;
uniform vec4 specular = vec4(1.0, 1.0, 1.0, 1.0);;

in Data {
	vec3 l_dir;
	vec3 normal;
	vec4 eye;
	vec2 texCoord;
	vec3 tangent;
} DataIn;

uniform float roughness = 0.04;
uniform float ka = 0.55;
uniform float ks = 0.35;

out vec4 colorOut;

void main() {

    vec3 t = normalize(DataIn.tangent);
	vec3 n = normalize(DataIn.normal);
	vec3 e = normalize(vec3(DataIn.eye));
	float intensity = max(dot(n,DataIn.l_dir),0.0);
	
	vec4 spec = vec4(0.0);
	if (intensity > 0) {
		float cos_eye = -t.t;
		float sin_eye = sqrt(1.0 - cos_eye* cos_eye);
		float cos_light = dot (t,DataIn.l_dir);
		float sin_light = sqrt(1.0 - cos_light * cos_light);
		float aniso = max(cos_light * cos_eye + sin_light*sin_eye, 0);
		float shad = max(dot(n,e), 0) * max(dot(n,DataIn.l_dir), 0.0);
		spec = specular * pow(aniso, 1/roughness) * shad;
	
	}
	colorOut = ka * diffuse*0.25 + ks * diffuse * intensity  + spec;

}