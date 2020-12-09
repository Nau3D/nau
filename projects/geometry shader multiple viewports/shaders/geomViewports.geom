#version 460
 
layout(triangles) in;
layout (triangle_strip, max_vertices=12) out;

uniform mat4 PV1, PV2, PV3, PV4;
uniform mat4 V1, V2, V3, V4;
uniform vec4 lightDir;

in DataV {
	vec3 normal;
} DataIn[];

out Data {
	vec3 normal;
	vec4 eye;
	vec4 lightDir;
} DataOut;

// in this example it is assumed that the model matrix is the identity matrix
// in order to simplify the shaders

void main() {

    mat4 V[4]; V[0] = V1;V[1] = V2;V[2] = V3;V[3] = V4;
    mat4 PV[4]; PV[0] = PV1; PV[1] = PV2; PV[2] = PV3; PV[3] =  PV4 ;

	for (int i = 0; i < 4; ++i)	{

		gl_ViewportIndex = i;

        // transform light to camera space
		vec4 ldCS = V[i] * lightDir;
        
		// first vertex
		gl_Position = PV[i] * gl_in[0].gl_Position;
		DataOut.normal = normalize(vec3(V[i] * vec4(DataIn[0].normal,0)));
		DataOut.eye = - V[i] *  gl_in[0].gl_Position;
		DataOut.lightDir = ldCS;
		EmitVertex();

		// second
		gl_Position = PV[i] * gl_in[1].gl_Position;
		DataOut.normal = normalize(vec3(V[i] * vec4(DataIn[1].normal,0)));
		DataOut.eye = - V[i] *  gl_in[1].gl_Position;
		DataOut.lightDir = ldCS;
		EmitVertex();

		// third vertex
		gl_Position = PV[i] * gl_in[2].gl_Position;
		DataOut.normal = normalize(vec3(V[i] * vec4(DataIn[2].normal,0)));
		DataOut.eye = - V[i] *  gl_in[2].gl_Position;
		DataOut.lightDir = ldCS;
		EmitVertex();

		EndPrimitive();
	}
}

