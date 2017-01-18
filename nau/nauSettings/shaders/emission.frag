#version 130

uniform vec4 emission;
uniform int texCount = 0;
uniform sampler2D texUnit;

in vec2 TexCoord;

out vec4 outColor;

void main()
{
	outColor = emission;
	if (texCount != 0)
		outColor *= texture(texUnit, TexCoord) * emission;
}
