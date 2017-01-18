#version 430

uniform mat4 PVM, V, M;
uniform mat3 NormalMatrix;
//uniform vec4 lightDirection;

in vec4 position;
in vec4 normal;
//in int centrals;

//out vec4 pos;
//out int centro;

out Data {
	vec4 pos;
	vec3 normal;
	int index;
} DataOut;


void main()
{
	//pos = vec4(V*M * position);
	DataOut.pos = V*M*position;
	DataOut.normal = normalize(NormalMatrix * vec3(normal));
	DataOut.index = gl_VertexID;
	gl_Position = PVM * position;
} 
