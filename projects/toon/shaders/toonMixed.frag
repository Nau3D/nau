#version 150

uniform	vec4 diffuse;

in float intensityV;

out vec4 outputF;

void main() {

	// compute the color based on the intensity
	if (intensityV > 0.90)
		outputF = diffuse;
	else if (intensityV > 0.75)
		outputF = 0.75 * diffuse;
	else if (intensityV > 0.5)
		outputF = 0.5 * diffuse;
	else if (intensityV > 0.25)
		outputF = 0.25 * diffuse;
	else
		outputF = 0.1 * diffuse;
}
