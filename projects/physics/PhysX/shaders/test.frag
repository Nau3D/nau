#version 420 

uniform vec4 lightSpecular;
uniform vec4 specular;
uniform float shininess;

in vec3 Normal;
in vec3 LightDirection;
in vec3 HalfVector;
in vec4 Diffuse;
in vec4 Ambient;

out vec4 outColor;

void main()
{
	vec3 n,halfV;
	float NdotL, NdotHV;
	
	/* The ambient term will always be present */
	vec4 color = Ambient;
	
	/* a fragment shader can't write a varying variable, hence we need
	a new variable to store the normalized interpolated normal */
	n = normalize(Normal);
	
	/* compute the dot product between normal and ldir */
	NdotL = max(dot(n,LightDirection),0.0);
	color.a = 0.4;
	if (NdotL > 0.0) {
		color += Diffuse * NdotL;
		halfV = normalize(HalfVector);
		NdotHV = max(dot(n,halfV),0.0);
		color += specular * 
				//lightSpecular * 
				vec4(1.0, 1.0, 1.0, 1.0) *
				pow(NdotHV, shininess);
		color.a = 0.99;
	}

	outColor = color;
}
