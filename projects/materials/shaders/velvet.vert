#version 330

uniform	mat4 mPVM;
uniform	mat4 mViewModel;
uniform mat4 mView;
uniform	mat3 mNormal;

uniform	vec4 lDir;	   // world space

in vec4 position;	// local space
in vec3 normal;		// local space


out Data {
	vec3 l_dir;
	vec3 normal;
	vec4 eye;
} DataOut;

void main () {
	
	DataOut.l_dir = normalize(vec3(mView * lDir));
	DataOut.normal = normalize(mNormal * normal);
	DataOut.eye = -(mViewModel * position);

	gl_Position = mPVM * position;	
}