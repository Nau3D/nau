#version 440

uniform mat4 PVM, V, VM,PV;

uniform mat3 NormalMatrix;

uniform vec4 cameraPosition, lightDirection, lightAmbient, diffuse, ambient;

uniform float shininess;

in vec4 normal;

in vec4 position;
layout(std430, binding = 1) buffer pswaterfall {
	vec4 pos[];
};

out vec3 Normal;
out vec3 LightDirection;
out vec3 HalfVector;
out vec4 Diffuse;
out vec4 Ambient;


void main() {	
	/* first transform the normal into eye space and 
	normalize the result */
	Normal = normalize(NormalMatrix * vec3(normal));
	
	
	vec4 p = position + pos[gl_InstanceID];
	p.w = 1.0;
	vec4 ResultPos = PV * p;
	
	
	/* now normalize the light's direction. Note that 
	according to the OpenGL specification, the light 
	is stored in eye space. Also since we're talking about 
	a directional light, the position field is actually direction */
	LightDirection = normalize(vec3(-lightDirection));

	/* Normalize the halfVector to pass it to the fragment shader */ //HV = normalize((EyePos-pos) - LDIR)
	//halfVector = normalize(gl_LightSource[0].halfVector.xyz);
	HalfVector = normalize(vec3(lightDirection + (cameraPosition - ResultPos)));
				
	/* Compute the diffuse, ambient and globalAmbient terms */
	// Diffuse = diffuse * lightDiffuse;
	Diffuse = diffuse * vec4(1.0, 1.0, 1.0, 1.0);
	Ambient = ambient * lightAmbient;
	Ambient += vec4(1.0) * ambient;

	gl_Position = ResultPos;
} 