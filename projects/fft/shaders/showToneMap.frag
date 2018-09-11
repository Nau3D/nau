#version 430

in vec2 texCoord;

uniform sampler2D lumiTex;
uniform int width;

out vec4 outColor;


void main() {
	
	// compute shifted coordinates to show (0,0) in the center
	ivec2 tc = ivec2(texCoord * vec2(width) + vec2(width/2)) ;
	tc.x = tc.x % width;
	tc.y = tc.y % width;
	
	// fetch the magnitude from level 0
	float lumi = texelFetch(lumiTex, tc, 0).r;
	
	//compute the maximum mipmap texture level
	int level = int(log2(width));		
	
	// fetch the maximum and minimum (result from the mipmap
	vec3 lumiAccum = texelFetch(lumiTex, ivec2(0), level).rgb;
	
	// compute the range of values in the texture
	float range = lumiAccum.g - lumiAccum.b;
	
	// map to [0,1] range
	float i = ((lumi - lumiAccum.b)/range);
	
	outColor = vec4(i,i,i, 1.0) ;
}	