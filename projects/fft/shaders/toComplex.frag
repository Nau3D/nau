#version 430

uniform sampler2D texUnit;

uniform int channels = 1;

in vec2 texCoord;

out vec2 complex;

void main () {
	
	vec3 texColor = texture(texUnit, texCoord).rgb;
	
	float luminance;
	// for black and white images (single channel)
	if (channels == 1)
		luminance = texColor.r;
	else
	// for color images - converto to luminance with same factors as DevIL 
	// https://github.com/DentonW/DevIL/blob/master/DevIL/src-IL/src/il_convert.cpp
		luminance = dot(vec3(0.212671, 0.715160, 0.072169), texColor);
		
	complex = vec2(luminance, 0);
}