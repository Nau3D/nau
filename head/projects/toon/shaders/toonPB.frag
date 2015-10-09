#version 150

layout (std140) uniform Material {
	vec4 diffuse;
};

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
	
	if (intensity > 0.90)
		outputF = diffuse;
	else if (intensity > 0.75)
		outputF = 0.75 * diffuse;
	else if (intensity > 0.5)
		outputF = 0.5 * diffuse;
	else if (intensity > 0.25)
		outputF = 0.25 * diffuse;
	else
		outputF = 0.1 * diffuse;
}
