#version 420 

layout (std140) uniform Material {
	vec4 diffuse;
	vec4 ambient;
	vec4 specular;
	vec4 emission;
	float shininess;
	int texCount;
};

uniform vec4 lightDirection, lightColor;
uniform sampler2D texUnit;

in vec3 Normal;
in vec2 TexCoord;
in vec3 LightDirection;
out vec4 outColor;

void main()
{
	vec4 color;
	float intensity;
	vec4 lightIntensityDiffuse;
	vec3 lightDir;
	vec3 n;
	
	if (texCount != 0 && texture(texUnit, TexCoord).a <= 0.25)
		discard;
		
	lightDir = -normalize(LightDirection);
	n = normalize(Normal);	
	intensity = max(dot(lightDir,n),0.0);
	
	lightIntensityDiffuse = lightColor * intensity;
	
	if (texCount == 0) {
		color = diffuse * lightIntensityDiffuse + diffuse * 0.3 ;
	}
	else {
		color = (diffuse * lightIntensityDiffuse + emission + 0.3) * texture(texUnit, TexCoord);
	}
	
	outColor = color;

		
}
