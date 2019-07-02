// http://www.renderman.org/RMR/Shaders/SHWShaders/SHW_velvet.sl

#version 330

uniform vec4 diffuse = vec4(0.15, 0.15, 0.75, 1.0);;


in Data {
	vec3 l_dir;
	vec3 normal;
	vec4 eye;
} DataIn;


uniform float backScatter = 0.3;
uniform float sheen = 0.75;
uniform float edginess = 10;
uniform float roughness = 0.2;
uniform float kd = 0.6;
uniform float ka = 0.3;
uniform float ks = 0.75;


out vec4 colorOut;


void main() {

	float shiny = 0.0;
	vec3 n = normalize(DataIn.normal);
	vec3 e = normalize(vec3(DataIn.eye));
	float intensity = max(dot(n,DataIn.l_dir),0.0);
	
	float cosine = max(dot(DataIn.l_dir, e), 0);
	shiny = pow(cosine, 1.0/roughness) * backScatter * sheen;

	cosine = max(dot(n, e),0) ;
	float sine = sqrt(1.0- sqrt(cosine));

	shiny += pow(sine, edginess) * intensity * sheen;

	colorOut = vec4(ka * diffuse + kd * intensity * diffuse + ks * shiny);
}