#version 420 

uniform vec4 lightDirection, lightColor;
uniform vec4 diffuse, ambient, emission;
uniform float shininess;

in vec3 Normal;
in vec3 LightDirection;
out vec4 outColor;

void main()
{
	vec4 color;
	float intensity;
	vec4 lightIntensityDiffuse;
	vec3 lightDir;
	vec3 n;
	
	lightDir = -normalize(LightDirection);
	n = normalize(Normal);	
	intensity = max(dot(lightDir,n),0.0);
		
	color = diffuse * lightColor * intensity + diffuse * 0.3 ;		
	outColor = color;

		
}
