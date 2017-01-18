#version 440

uniform vec4 lightDirection;
uniform mat4 lightSpaceMat1,lightSpaceMat2,lightSpaceMat3,lightSpaceMat4;
uniform mat4 PVM;
uniform mat4 V, M;
uniform mat3 NormalMatrix;

in vec4 position;
in vec4 normal;
in vec4 texCoord;

out vec4 viewSpacePos;
out vec4 projShadowCoord[4];
out vec3 normalV, lightDir;


void main()
{
	normalV = normalize (NormalMatrix * vec3(normal));
	
	lightDir = normalize (vec3(V * -lightDirection));

/* 	for (int i = 0; i < 4; i++) {
		projShadowCoord[0] = lightSpaceMat1 * M * position;
		//projShadowCoord[i] = projShadowCoord[i] / projShadowCoord[i].w;
	}
 */		projShadowCoord[0] = lightSpaceMat1 * M * position;
		projShadowCoord[1] = lightSpaceMat2 * M * position;
		projShadowCoord[2] = lightSpaceMat3 * M * position;
		projShadowCoord[3] = lightSpaceMat4 * M * position;

	viewSpacePos = V * M * position;
	
	gl_Position = PVM * position;
}  