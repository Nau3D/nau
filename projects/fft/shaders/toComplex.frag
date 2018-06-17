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
	// for color images
	//https://en.wikipedia.org/wiki/Relative_luminance
		luminance = dot(vec3(0.2126, 0.7152, 0.0722), texColor);
		
	complex = vec2(luminance, 0);
}