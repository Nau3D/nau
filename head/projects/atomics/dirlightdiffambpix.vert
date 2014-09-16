#version 420

uniform mat4 PVM,V;
uniform mat3 NormalMatrix;

uniform vec4 lightDirection;

in vec4 position;
in vec4 normal;
in vec4 texCoord0;

out vec4 vertexPos;
out vec2 TexCoord;
out vec3 Normal;
out vec3 LightDirection;

void main()
{
	Normal = normalize(NormalMatrix * vec3(normal));
	LightDirection = normalize(vec3(V * lightDirection));
	TexCoord = vec2(texCoord0);
	gl_Position = PVM * position;
}