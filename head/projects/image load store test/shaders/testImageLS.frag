#version 430

uniform writeonly image2D imageUnit;

in vec3 Normal;

out vec4 outColor;

void main()
{
	float intensity;
	vec3 lightDir;
	vec3 n;
			
	lightDir = normalize(vec3(1,-1,1));
	n = normalize(Normal);	
	intensity = max(dot(lightDir,n),0.0);
	
	imageStore(imageUnit, ivec2(gl_FragCoord.xy),vec4(1.0,1.0,0.0,1.0));
	outColor = vec4(0.0,1.0,0.0,1.0);
	//discard;
}
