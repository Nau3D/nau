#version 330

uniform sampler2D diffuseY, specularMap, rust, noise;
uniform float rusting;
uniform float shininess = 80;

in Data {
	vec3 eye;
	vec2 texCoord;
	vec3 l_dir;
	vec3 normal;
} DataIn;

out vec4 colorOut;


void main() {

	// normalize vectors
	vec3 normal = normalize(DataIn.normal);
	vec3 eye = normalize(DataIn.eye);
	vec3 l_dir = normalize(DataIn.l_dir);
		
	// get texture data
	vec4 texColor = texture(diffuseY, DataIn.texCoord);
	vec4 texRust = texture(rust, DataIn.texCoord);
	vec4 texSpecular = texture(specularMap, DataIn.texCoord);
	float noise = texture(noise, DataIn.texCoord).r;
	
	// mix rust with diffuse
	float f = smoothstep(rusting, rusting+0.05, noise);
	vec4 color = mix(texColor, texRust, f);
	
	// set the specular term to black
	vec4 spec = vec4(0.0);


	float intensity = max(dot(normal,l_dir), 0.0);

	// if the vertex is lit and there is little or no rust
	// compute the specular color
	if (intensity > 0.0 && f < rusting+0.05) {
		// compute the half vector
		vec3 h = normalize(l_dir + eye);	
		// compute the specular intensity
		float intSpec = max(dot(h,normal), 0.0);
		// compute the specular term into spec
		spec = texSpecular * pow(intSpec,shininess);
	}
	colorOut = max(intensity * color + spec +  color * 0.5,0);

}

