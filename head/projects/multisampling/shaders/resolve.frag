#version 420

uniform sampler2DMS texUnit;
uniform int samples = 16;

in Data {
	vec4 texCoord;
} DataIn;

out vec4 outputF;

void main()
{
	int samples=8;
	ivec2 tc = ivec2(DataIn.texCoord.xy * 1024);

	outputF = vec4(0.0);
	for (int i = 0; i < samples; ++i) {
		outputF += texelFetch(texUnit, tc ,i) ;
	}
	outputF = outputF/samples;
	//outputF =  texelFetch(texUnit, tc ,0);
	//outputF = vec4(1.0,0.0,0.0,1.0);
} 
