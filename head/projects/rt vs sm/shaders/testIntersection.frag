#version 420

//layout(binding=1, offset=8) uniform atomic_uint at2;

uniform mat4 lightSpaceMat;
uniform sampler2D vertexA, vertexB, vertexC, positions;
uniform vec4 lightDir;



in vec2 texCoordV;
out vec4 outColor;

void main() {

	vec4 pos = texture(positions, texCoordV);
	vec2 p = vec2(lightSpaceMat * vec4(vec3(pos),1.0f));
	vec3 vA = texture(vertexA, p).xyz;
	vec3 vB = texture(vertexB, p).xyz;
	vec3 vC = texture(vertexC, p).xyz;
	
	vec4 color;
	vec3 ld = vec3(-lightDir);
	
	vec3 e1 = vB - vA;
	vec3 e2 = vC - vA;

	if (length(e1) == 0.0 || length(e2) == 0.0) {
//		outColor = pos;
		outColor = vec4(0.0, 1.0, 0.0, 1.0);
		return;
	}
	
	vec3 h = cross(ld,e2);
	float a = dot(e1,h);
	
	float f,u,v,t;
	vec3 s,q;

	if (a > -0.00001 && a < 0.00001)
	{
		color = vec4(0.0, 1.0, 0.0, 1.0);
//		color = pos;
	}
	else
	{
		f = 1.0/a;
		s = vec3(pos) - vA;
		u = f * (dot(s,h));

		if (u < 0.0 || u > 1.0)
		{
			color = vec4(0.0, 1.0, 0.0, 1.0);
//			 color = pos;
		}
		else
		{
			q = cross(s,e1);
			v = f * (dot(ld,q));

			if (v < 0.0 || u + v > 1.0)
			{
				color = vec4(0.0, 1.0, 0.0, 1.0);
//				color = pos;
			}
			else
			{
				t = f * (dot(e2,q));

				if (t > 0.00001)
				{
//					if (pos.w == 1.0)
//						atomicCounterIncrement(at2);

					pos.w = 0.0;
					color = vec4(1.0,0.0,0.0,0.0);
				}
				else
				{
					color = vec4(0.0, 1.0, 0.0, 1.0);
//					color = pos;
				}
			}
		}
	}
	
	

	outColor = color;

}