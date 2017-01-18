#version 330

uniform vec4 lightDirection, lightColor;
uniform vec4 diffuse, ambient, emission;
uniform float shininess;
uniform int texCount;
uniform sampler2D texUnit;

in vec3 Normal;
in vec2 TexCoord;
in vec3 LightDirection;
out vec4 outColor;

void main()
{
	vec4 color;
	vec4 amb;
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
	float alpha;
	if (texCount == 0) {
		color = diffuse * lightIntensityDiffuse + diffuse * 0.3 + emission ;
		alpha = diffuse.a;
	}
	else {
		color = (diffuse * lightIntensityDiffuse + emission + 0.3) * texture(texUnit, TexCoord);
		alpha = texture(texUnit, TexCoord).a * diffuse.a;
	}
	outColor = vec4(vec3(color ), alpha);
//	output = (color * X) + amb;
//	outColor = vec4(1,0,0,0);

}
