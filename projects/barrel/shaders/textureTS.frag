#version 330

uniform sampler2D diffuse, normalMap, specularMap, rust, noise;
uniform float rusting;
uniform float shininess = 80;

in Data {
	vec3 eye;
	vec2 texCoord;
	vec3 l_dir;
} DataIn;

out vec4 colorOut;

void main() {

	// get texture color
	vec4 texColor = texture(diffuse, DataIn.texCoord);
	vec4 texRust = texture(rust, DataIn.texCoord);
	vec4 texNoise = texture(noise, DataIn.texCoord);
	vec4 texSpecular = texture(specularMap, DataIn.texCoord);
	vec3 texNormal = normalize(vec3(texture(normalMap, DataIn.texCoord) * 2.0 - 1.0));
	
	// mix rust with diffuse
	float f = smoothstep(rusting, rusting + 0.05, texNoise.r);
	vec4 color = mix(texColor, texRust, f);
	
	// set the specular term to black
	vec4 spec = vec4(0.0);

	// normalize both input vectors
	vec3 e = normalize(vec3(DataIn.eye));
	vec3 l = normalize(DataIn.l_dir);
	float intensity = max(dot(texNormal, l), 0.0);

	// if the vertex is lit compute the specular color
	if (intensity > 0.0 && f < rusting + 0.05) {
		// compute the half vector
		vec3 h = normalize(l + e);	
		// compute the specular intensity
		float intSpec = max(dot(h,texNormal), 0.0);
		// compute the specular term into spec
		spec = texSpecular * pow(intSpec,shininess);
	}
	colorOut = max(intensity * color + spec +  color * 0.5,0);
}