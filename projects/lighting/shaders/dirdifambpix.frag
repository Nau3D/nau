#version 150

uniform	vec4 diffuse;

in Data{
	vec3 normal;
	vec3 l_dir;
};

out vec4 outputF;

void main()
{
	float intensity;
	vec3 l, n;
	
	n = normalize(normal);
	// no need to normalize the light direction!
	intensity = max(dot(l_dir,n),0.0);
	
	outputF = max(diffuse * 0.25, diffuse * intensity);
}