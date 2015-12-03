#version 330

uniform vec4 lightDirection;
uniform mat4 lightSpaceMat;

uniform mat4 PVM;
uniform mat4 V, M;
uniform mat3 NormalMatrix;

in vec4 position;
in vec4 normal;
in vec4 texCoord0;

out vec4 projShadowCoord;
out vec3 normalV;
out vec2 texCoordV;
out vec3 lightDir;
out vec4 pos;


void main() 
{
	normalV = normalize (NormalMatrix * vec3(normal));
	texCoordV = vec2(texCoord0);
	lightDir = normalize (vec3(V * -lightDirection)) ;
			
	projShadowCoord = lightSpaceMat * M * position;
	pos = position;
	gl_Position = PVM * position;
} 
