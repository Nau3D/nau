#version 330

uniform	mat4 m_pvm;

uniform	vec4 l_dir;	// global space

in vec4 position;	// local space


void main () {

	int index = gl_InstanceID;
	
	int h = index / 10000;
	index -= h*10000;
	int col = index % 100;
	int row = index / 100;
	
	vec4 pos = position + vec4(col * 2 * 0.68, h*2, row * 2 * 0.68, 1);
	gl_Position = m_pvm * pos;	
}
