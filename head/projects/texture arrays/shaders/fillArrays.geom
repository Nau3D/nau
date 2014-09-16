#version 330

layout(triangles) in;
layout (triangle_strip, max_vertices=6) out;

uniform mat4 P,M,V;
uniform vec3 camPos;
uniform vec3 camView, camUp;
uniform vec3 lightDirection;

in vec2 TexCoordv[];
in vec3 Normalv[];


out vec2 TexCoord;
out vec3 Normal;
out vec3 LightDirection;


void main() {

	vec3 right = normalize(cross(camView, camUp));
	vec3 up = cross(right, camView);
	
	mat4 v1 = mat4(vec4(right,0), vec4(up,0), vec4(-camView, 0), vec4(0,0,0,1));
	v1 = transpose(v1);
	
	gl_Layer = 0;
	vec3 pos = camPos + 0.05 * right;
	mat4 v2 = mat4(vec4(1,0,0,0), vec4(0,1,0,0), vec4(0,0,1,0), vec4(-pos, 1));
	//v2 = transpose(v2);
	
	mat4 v = v1 * v2;
	mat4 n = v * M;
	for(int i = 0; i < gl_in.length(); ++i)
	{
		 // copy position
		gl_Position = P * (n * gl_in[i].gl_Position);
		Normal = normalize(vec3(n * vec4(Normalv[i],0)));
		TexCoord = TexCoordv[i];
		LightDirection = vec3(n * vec4(lightDirection,0));
		// done with the vertex
		EmitVertex();
	}
	EndPrimitive();

	gl_Layer = 1;
	pos = camPos - 0.05 * right;
	v2 = mat4(vec4(1,0,0,0), vec4(0,1,0,0), vec4(0,0,1,0), vec4(-pos, 1));
	
	v = v1 * v2;
	n = v * M;
	for(int i = 0; i < gl_in.length(); ++i)
	{
		 // copy position
		gl_Position = P * (n * gl_in[i].gl_Position);
		Normal = vec3(n * vec4(Normalv[i],0));
		TexCoord = TexCoordv[i];
		LightDirection = vec3(n * vec4(lightDirection,0));
		// done with the vertex
		EmitVertex();
	}
	EndPrimitive();

	
}