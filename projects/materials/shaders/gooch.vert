#version 330

uniform	mat4 mPVM;
uniform	mat4 mViewModel;
uniform mat4 mView;
uniform	mat3 mNormal;

uniform	vec4 lDir;	   // world space

in vec4 position;	// local space
in vec3 normal;		// local space

// the data to be sent to the fragment shader
out vec3 lDirV; // camera space
out vec3 normalV; // camera space
out vec3 eyeV;


void main () {
	
	normalV = normalize(mNormal * normal);
	eyeV = -vec3(mViewModel * position);
	lDirV = normalize(-vec3(mView * lDir));
	gl_Position = mPVM * position;	
}