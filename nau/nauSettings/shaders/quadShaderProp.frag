#version 130

uniform sampler2D texUnit;
uniform vec2 viewport = vec2(512,1024);

in vec2 texCoordV;

out vec4 outColor;

void main() {

	vec2 delta;
	vec2 ts = vec2(textureSize(texUnit,0));
	vec2 v = ts/vec2(512,1024);
	if (v.x > v.y) {
		v.y = v.x/v.y;
		v.x = 1.0;
		delta = vec2(0.0, -v.y*0.5+0.5);
//
	}
	else {
		v.x = v.y/v.x;
		v.y = 1.0;
		delta = vec2(-v.x*0.5 + 0.5,0.0);
	}
	vec2 coord = 	texCoordV * v + delta;
//	if (coord.x >= 0 && coord.y >= 0 && coord.x < 1 && coord.y < 1)
	if (coord.x >= 0 && coord.y >= 0 && coord.x < 1 && coord.y < 1)
		outColor = texture(texUnit, coord);
	else
		outColor = vec4(1,1,1,1);
	//outColor = vec4(v.y,0,0,1);
}