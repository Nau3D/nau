#version 330

uniform	mat4 mPVM;
uniform	mat4 mViewModel;
uniform mat4 mView;
uniform	mat3 mNormal;

uniform	vec4 lDir;	   // world space

in vec4 position;	// local space
in vec3 normal;		// local space
in vec4 tangent;	// local space
in vec2 texCoord0;		// local space

out Data {
	vec3 l_dir;
	vec3 normal;
	vec4 eye;
	vec2 texCoord;
	vec3 tangent;
} DataOut;

void main () {
	
	DataOut.l_dir = normalize(vec3(mView * lDir));
	DataOut.texCoord = texCoord0;
	DataOut.normal = normalize(mNormal * normal);
	DataOut.eye = -(mViewModel * position);
	DataOut.tangent = normalize(vec3(mViewModel * tangent));

	gl_Position = mPVM * position;	
}