#version 330

uniform sampler2D texUnit;
uniform float dist;
uniform float fogMin,fogMax;
uniform vec4 fogColor;

in vec3 normalV;
in vec2 texCoordV;
in vec3 lightDirV;
in float depth;

out vec4 colorOut;

void main() {

	float intensity = max(0.0, dot(normalize(normalV), lightDirV ));
	vec4 texColor = texture(texUnit, texCoordV);
	if (depth > dist)
		colorOut = vec4(intensity + 0.2);
	else
		colorOut = texColor * (intensity + 0.2);
		
	float fogFactor = (fogMax - depth) / 
				(fogMax - fogMin);
	fogFactor = clamp( fogFactor, 0.0, 1.0 );
	colorOut = mix( fogColor, colorOut, fogFactor );
	//colorOut = fogColor;//vec4(lightDirV, 0);
}