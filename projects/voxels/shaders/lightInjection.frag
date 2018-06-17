#version 440

out vec4 FragColor;

layout (rgba8) uniform coherent volatile image3D grid;

uniform int GridSize;

uniform sampler2D texRSM, texPos;

in vec2 texCoordV;

void main()
{
	// retrieve data from RSM pass
	vec4 pos = texture(texPos, texCoordV);
	vec4 color = texture(texRSM, texCoordV);
	
	// compute grid cell coordinates
	ivec3 coord = ivec3((pos*0.5 + 0.5)*GridSize);

	// update cell color 
	if (color.xyz != vec3(0,0,0))
		imageStore(grid, coord, vec4(color.xyz,1));
}