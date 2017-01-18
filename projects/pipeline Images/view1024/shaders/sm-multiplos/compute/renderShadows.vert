#version 430

uniform mat4 PVM, V, M;
uniform mat3 NormalMatrix;
uniform vec4 lightDirection;
uniform mat4 lightSpaceMat;

in vec3 normal;
in vec4 position;
in vec4 texCoord0;

out vec3 texCoordV;
out vec4 viewSpacePos;
out vec4 modelPos;
out vec4 projShadowCoord;
out vec3 normalV, lightDir;

void main() {

	normalV = normalize (NormalMatrix * vec3(normal));
	
	lightDir = normalize (vec3(V * -lightDirection));

	projShadowCoord = lightSpaceMat * M * position;

	texCoordV = vec3(texCoord0); 

	viewSpacePos = V * M * position;
	
	modelPos = M * position;
	
	gl_Position = PVM * position;


/*
 	normalV = normalize (NormalMatrix * vec3(normal));
	//lightDir = normalize (vec3(V * -lightDirection)) ;
	eye = -(V * M * position);
	pointFromLight = lightSpaceMat * M * position;
	
	
	gl_Position = PVM * position;
	
	//gl_Position = position;
	*/	
}