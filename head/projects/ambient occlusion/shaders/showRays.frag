#version 430

in flat int hit;

out vec4 color;

void main(void) {

	 if (hit > 0)
		 color = vec4(1,0,0,1);
	 else
		color = vec4(0,1,0,1);
}
