#version 430

uniform sampler2D texUnit;
uniform writeonly image2D imageUnit;

in vec4 texCoord;

out vec4 outColor;

void main()
{
	vec4 c = texture(texUnit, texCoord.xy);
	imageStore(imageUnit, ivec2(gl_FragCoord.xy),c);
	outColor = c;
}
