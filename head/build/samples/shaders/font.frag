#version 330


uniform vec4 emission;
uniform sampler2D texUnit;

in vec2 TexCoord;

out vec4 outColor;

void main()
{
	vec4 c = texture(texUnit, TexCoord);
	outColor = emission * c;
}
