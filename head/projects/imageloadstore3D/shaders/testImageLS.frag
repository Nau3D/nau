#version 440

layout (r32ui) uniform coherent uimage3D imageUnit;

layout(std430, binding = 2) buffer test {
	unsigned int k[];
};

layout(binding=1, offset=0)  uniform atomic_uint at0;
layout(binding=1, offset=4)  uniform atomic_uint at1;
layout(binding=1, offset=8)  uniform atomic_uint at2;
layout(binding=1, offset=12) uniform atomic_uint at3;
layout(binding=1, offset=16) uniform atomic_uint at4;
layout(binding=1, offset=20) uniform atomic_uint at5;
layout(binding=1, offset=24) uniform atomic_uint at6;
layout(binding=1, offset=28) uniform atomic_uint at7;

in vec3 Normal;
in vec4 Position;

out vec4 outColor;

void main()
{
	float intensity;
	vec3 lightDir;
	vec3 n;
			
	lightDir = normalize(vec3(1,-1,1));
	n = normalize(Normal);	
	intensity = max(dot(lightDir,n),0.0);
	
	ivec3 v = ivec3(0);
	if (Position.x < 0 && Position.y < 0 && Position.z < 0) {
		atomicCounterIncrement(at0);
		atomicAdd(k[0], 1);
	}
	else if (Position.x < 0 && Position.y < 0 && Position.z >= 0) {
		v.z = 1;
		atomicCounterIncrement(at1);
		atomicAdd(k[1], 1);
	}
	else if (Position.x < 0 && Position.y >= 0 && Position.z < 0) {
		v.y = 1;
		atomicCounterIncrement(at2);
		atomicAdd(k[2], 1);
	}
	else if (Position.x < 0 && Position.y >= 0 && Position.z >= 0) {
		v.y = 1; 
		v.z = 1;
		atomicCounterIncrement(at3);
		atomicAdd(k[3], 1);
	}
	else if (Position.x >= 0 && Position.y < 0 && Position.z < 0) {
		v.x = 1;
		atomicCounterIncrement(at4);
		atomicAdd(k[4], 1);
	}
	else if (Position.x >=0 && Position.y < 0 && Position.z >= 0) {
		v.x = 1;
		v.z = 1;
		atomicCounterIncrement(at5);
		atomicAdd(k[5], 1);
	}
	else if (Position.x >= 0 && Position.y >= 0 && Position.z < 0) {
		v.x = 1;
		v.y = 1;
		atomicCounterIncrement(at6);
		atomicAdd(k[6], 1);
	}
	else if (Position.x >= 0 && Position.y >= 0 && Position.z >= 0) {
		v.x = 1;
		v.y = 1; 
		v.z = 1;
		atomicCounterIncrement(at7);
		atomicAdd(k[7], 1);
	}

	imageAtomicAdd(imageUnit, v, uint(1));
	k[0] = imageSize(imageUnit).x;
	k[1] = imageSize(imageUnit).y;
	k[2] = imageSize(imageUnit).z;
	
	outColor = vec4(0.0,1.0,0.0,1.0) * intensity;
}
