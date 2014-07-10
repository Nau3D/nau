#version 410
 
layout(triangles) in;
layout (triangle_strip, max_vertices=3) out;
 
uniform mat4 PVM;

flat out vec4 vertexA, vertexB, vertexC;
 

 void main()
{

	vertexA = gl_in[0].gl_Position;
	vertexB = gl_in[1].gl_Position;
	vertexC = gl_in[2].gl_Position;
	
	gl_Position = PVM * gl_in[0].gl_Position;
	EmitVertex();

	gl_Position = PVM * gl_in[1].gl_Position;
	EmitVertex();

	gl_Position = PVM * gl_in[2].gl_Position;
	EmitVertex();
}
