#version 330

uniform sampler2D t0,t1,t2,t3,t4,t5;

in vec3 lDirV; // camera space
in vec3 normalV; // camera space
in vec2 texCoordV;

out vec4 colorOut;

void main() {
	
	vec4 color1, color2;
	float mixFactor;

	// gv normalize both input vectors
	vec3 n = normalize(normalV);

	// intensity goes from -1 to 1
	float intensity = max(0.0, dot(n,lDirV));
	
	float interval = 1.0/6.0;
	if (intensity < interval) {
		color1 = texture(t0, texCoordV);
		color2 = texture(t1, texCoordV);
		mixFactor = intensity * 6;		
	}
	else if (intensity < interval*2) {
		color1 = texture(t1, texCoordV);
		color2 = texture(t2, texCoordV);
		mixFactor = (intensity - interval)* 6;		
	}
	else if (intensity < interval*3) {
		color1 = texture(t2, texCoordV);
		color2 = texture(t3, texCoordV);
		mixFactor = (intensity - interval*2)* 6;		
	}
	else if (intensity < interval*4) {
		color1 = texture(t3, texCoordV);
		color2 = texture(t4, texCoordV);
		mixFactor = (intensity - interval*3)* 6;		
	}
	else if (intensity < interval*5) {
		color1 = texture(t4, texCoordV);
		color2 = texture(t5, texCoordV);
		mixFactor = (intensity - interval*4)* 6;		
	}
	else {
		color1 = texture(t5, texCoordV);
		color2 = vec4(1);
		mixFactor = (intensity - interval*5)* 6;		
	}
	
	colorOut = mix(color1, color2, mixFactor);
}