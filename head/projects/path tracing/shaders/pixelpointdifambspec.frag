#version 420

uniform sampler2D texUnit;
uniform int texCount;
uniform vec4 diffuse;
uniform vec4 specular;

in vec3 normalV;
in vec3 eyeV;
in vec3 lightDirV;
in vec2 texCoordV;

layout (location = 0) out vec4 colorOut0;
layout (location = 1) out vec4 colorOut1;

void main() {

	// vec4 dif;
	// vec4 spec = vec4(0.0);

	// vec3 n = normalize(normalV);
	// vec3 lDir = normalize(vec3(lightDirV));
	// float intensity = max(dot(n,lDir), 0.0);


	// if (intensity > 0) {
		// vec3 h = normalize(lDir + normalize(eyeV));	
		// float intSpec = max(dot(h, n), 0.0);
		// spec = specular * pow(intSpec, 100);
	// }
	// dif = max(diffuse*0.5, diffuse * intensity);
	
	// if (texCount != 0)
		// dif *= texture(texUnit, texCoordV);
		
	// colorOut0 = (intensity + 0.2) * dif + spec ;
	//colorOut = vec4(normalV,0);
	
	
	if (texCount == 0)
		colorOut0 = diffuse;
	else
		colorOut0 = texture(texUnit, texCoordV);
	colorOut1 = colorOut0;
}