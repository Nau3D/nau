#version 330


in vec4 position;
in vec4 texCoord0;

uniform vec4 lightDir;
uniform mat4 m_view;

out vec2 texCoordV;
out vec3 light_Dir;

void main() {

	light_Dir = normalize(vec3(-m_view * lightDir));
	gl_Position = position;
	texCoordV = vec2(texCoord0);	
}