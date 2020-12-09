#version 460

in vec4 position;
in vec3 normal;

out DataV {
	vec3 normal;
} DataOut;

void main()
{
    DataOut.normal = normal;
	gl_Position = position; 
}