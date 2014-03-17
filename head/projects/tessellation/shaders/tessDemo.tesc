#version 410

layout(vertices = 16) out;

in vec4 posV[];
out vec4 posTC[];

uniform float olevel = 8.0, ilevel= 8.0;

void main() {

	posTC[gl_InvocationID] = posV[gl_InvocationID];
	
	if (gl_InvocationID == 0) {
		gl_TessLevelOuter[0] = 8;//olevel;
		gl_TessLevelOuter[1] = 8;//olevel;
		gl_TessLevelOuter[2] = 8;//olevel;
		gl_TessLevelOuter[3] = 8;//olevel;
		gl_TessLevelInner[0] = 8;//ilevel;
		gl_TessLevelInner[1] = 8;//ilevel;
	}
}