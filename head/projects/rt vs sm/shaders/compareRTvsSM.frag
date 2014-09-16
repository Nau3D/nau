#version 420

uniform sampler2D rayTrace, shadowMap, projShadowContour, intersectionRes;
layout(binding=1, offset=0) uniform atomic_uint at;
layout(binding=1, offset=4) uniform atomic_uint at1;
layout(binding=1, offset=8) uniform atomic_uint at2;
layout(binding=1, offset=12) uniform atomic_uint at3;
layout(binding=1, offset=16) uniform atomic_uint at4;
layout(binding=1, offset=20) uniform atomic_uint at5;
layout(binding=1, offset=24) uniform atomic_uint at6;
layout(binding=1, offset=28) uniform atomic_uint at7;
layout(binding=1, offset=32) uniform atomic_uint at8;
layout(binding=1, offset=36) uniform atomic_uint at9;
layout(binding=1, offset=40) uniform atomic_uint at10;

in vec2 texCoordV;

out vec4 outColor;

void main() {
	
	float i = texture(intersectionRes, texCoordV).r;
	float r = texture(rayTrace, texCoordV).r;
	vec4 s = texture(shadowMap, texCoordV);
	float p = texture(projShadowContour, texCoordV).r;
	
	// count total difference between shadow map and ray trace
	if (r != s.x)
		atomicCounterIncrement(at);

	// count incorrect pixels (light RT - shadow SM)
	if (r == 1.0 && s.x == 0.0)
		atomicCounterIncrement(at1);

	// count incorrect pixels (shadow RT - light SM)
	if (r == 0.0 && s.x == 1.0)
		atomicCounterIncrement(at2);

	// pixels not facing the light
	if (s.y == 0.0)	
		atomicCounterIncrement(at3);	

	// check how many pixels are covered by the shadow map contours
	if (p != 0.0)
		atomicCounterIncrement(at4);

	// from those, how many are facing the light
	if (p != 0 && s.y != 0.0)
		atomicCounterIncrement(at5);

	// check how many pixels do intersect triangle stored in shadow map
	if (i == 1.0)
		atomicCounterIncrement(at6);

	// count incorrect pixels in light (SM) that get corrected
	if (i == 1.0 && r == 0.0 && s.x == 1.0)
		atomicCounterIncrement(at7);

	// count correct pixels in shadow (SM) that get confirmed
	if (i == 1 && r == 0.0 && s.x == 0.0)
		atomicCounterIncrement(at8);

	// count pixels where projection and intersection occurs
	if (i == 1 && p != 0 && s.y != 0.0)
		atomicCounterIncrement(at9);

	// wrong pixels not covered by shadow map contour proj
	if (s.y != 0.0 && r != s.x && p == 0)
		atomicCounterIncrement(at10);
	
	if (i == 1 || s.y == 0 || (r == 0 && s.x == 0)) {
		outColor = vec4(0.0);
	}
	else if (s.x == 1 && r == 1)
		outColor = vec4(1.0);
	else if (r == 0 && s.x != 0 && p != 0)
		outColor = vec4(0.0, 1.0, 0.0, 1.0);
	else if (r == 1 && s.x == 0 && p != 0)
		outColor = vec4(1.0, 0.0, 0.0, 1.0);
	else {
		outColor = vec4(0.0, 0.0, 1.0, 1.0);
	}
}		
