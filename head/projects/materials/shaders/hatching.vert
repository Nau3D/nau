#version 330

uniform	mat4 mPVM;
uniform	mat4 mViewModel;
uniform mat4 mView;
uniform	mat3 mNormal;

uniform	vec4 lDir;	   // world space

in vec4 position;	// local space
in vec3 normal;		// local space
in vec2 texCoord0;

// the data to be sent to the fragment shader
out vec3 lDirV; // camera space
out vec3 normalV; // camera space
out vec2 texCoordV;


void main () {
	
	normalV = normalize(mNormal * normal);
	texCoordV = texCoord0;
	lDirV = normalize(-vec3(mView * lDir));
	gl_Position = mPVM * position;	
}