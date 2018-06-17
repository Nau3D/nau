#version 330

uniform sampler2D normal, texCoord, tangent, 
	bitangent, normalMap, specularMap,eye, color;
uniform float shininess = 80;

in vec2 texCoordV;
in vec3 light_Dir;

out vec4 outColor;

void main() {

	// get texture data
	vec3 n = texture(normal, texCoordV).xyz * 2 - 1;
	if (n == vec3(1, 1, 1))
		discard;
	vec3 t = texture(tangent, texCoordV).xyz * 2 - 1;  
	vec2 tc = texture(texCoord, texCoordV).xy;   
	vec3 e = normalize(texture(eye, texCoordV).xyz);   
	vec4 thecolor = texture(color, texCoordV);
	vec4 texSpecular = texture(specularMap, tc);
	vec3 texNormal = normalize(vec3(texture(normalMap, tc) * 2.0 - 1.0));
	
	// compute bitangent
	vec3 bt = cross(n,t);
	
	// compute tbn matrix
	mat3 tnb = mat3(t, bt, n);
	
	// set the specular term to black
	vec4 spec = vec4(0.0);

	// normalize both input vectors
	vec3 nn = normalize(tnb * texNormal);

	float intensity = max(dot(nn,light_Dir), 0.0);

	// if the vertex is lit and there is little or no rust
	// compute the specular color
	if (intensity > 0.0 && thecolor.w < 0.6) {
		// compute the half vector
		vec3 h = normalize(light_Dir + e);	
		// compute the specular intensity
		float intSpec = max(dot(h,nn), 0.0);
		// compute the specular term into spec
		spec = texSpecular * pow(intSpec,shininess);
	}
	outColor = max(intensity * thecolor + spec +  thecolor * 0.5,0);
}


