#version 420 

layout(binding=2)  uniform atomic_uint at0;
layout(binding=2, offset=4) uniform atomic_uint at1;
layout(binding=2, offset=8) uniform atomic_uint at2;
layout(binding=2, offset=12) uniform atomic_uint at3;
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
	
	float m = max(max(color.r, color.g),color.b);
	
	if (m == color.r) {
		atomicCounterIncrement(at0);
	}
	else if (m == color.g) {
		 atomicCounterIncrement(at1);
	 }
     else
		 atomicCounterIncrement(at2);
	atomicCounterIncrement(at3);
	memoryBarrier(); 
	outColor = color;

		
}
