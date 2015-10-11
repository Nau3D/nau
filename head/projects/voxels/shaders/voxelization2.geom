#version 440

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 camXPV, camYPV, camZPV;
uniform vec2 WindowSize;

in vec3 normalV[3];
in vec2 texCoordV[3];

out vec3 worldPos;
out vec4 bBox;

float pixelDiagonal = 1.0/WindowSize.x;

void expandTriangle(inout vec4 screenPos[3]) {

	vec2 side0N = normalize(screenPos[1].xy - screenPos[0].xy);
	vec2 side1N = normalize(screenPos[2].xy - screenPos[1].xy);
	vec2 side2N = normalize(screenPos[0].xy - screenPos[2].xy);
	screenPos[0].xy = screenPos[0].xy + normalize(-side0N+side2N)*pixelDiagonal;
	screenPos[1].xy = screenPos[1].xy + normalize(side0N-side1N)*pixelDiagonal;
	screenPos[2].xy = screenPos[2].xy + normalize(side1N-side2N)*pixelDiagonal;
}


void main() {

	vec4 screenPos[3];
	vec3 p1 = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	vec3 p2 = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	vec3 normal = abs(cross(p2,p1));

	float m = max(normal.z, max(normal.y, normal.x));

	if (m == normal.x) {
		screenPos[0] = camXPV * gl_in[0].gl_Position;
		screenPos[1] = camXPV * gl_in[1].gl_Position;
		screenPos[2] = camXPV * gl_in[2].gl_Position;
	}
	else if (m == normal.y) {
		screenPos[0] = camYPV * gl_in[0].gl_Position;
		screenPos[1] = camYPV * gl_in[1].gl_Position;
		screenPos[2] = camYPV * gl_in[2].gl_Position;
	}
	else /*if (m == normal.z)*/ {
		screenPos[0] = camZPV * gl_in[0].gl_Position;
		screenPos[1] = camZPV * gl_in[1].gl_Position;
		screenPos[2] = camZPV * gl_in[2].gl_Position;
	}
	
	// Calculate screen space bounding box to be used for clipping in the fragment shader.
	bBox.xy = min(screenPos[0].xy, min(screenPos[1].xy, screenPos[2].xy));
	bBox.zw = max(screenPos[0].xy, max(screenPos[1].xy, screenPos[2].xy));
	bBox.xy -= vec2(pixelDiagonal);
	bBox.zw += vec2(pixelDiagonal);
	
	// Expand triangle for conservative rasterization.
	expandTriangle(screenPos);
	
	worldPos = vec3(gl_in[0].gl_Position);
	gl_Position = screenPos[0];
	EmitVertex();
	
	worldPos = vec3(gl_in[1].gl_Position);
	gl_Position = screenPos[1];
	EmitVertex();

	worldPos = vec3(gl_in[2].gl_Position);
	gl_Position = screenPos[2];
	EmitVertex();
	
	EndPrimitive();
}
