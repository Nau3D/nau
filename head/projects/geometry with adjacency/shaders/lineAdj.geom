#version 420
 
layout(triangles_adjacency) in;
layout (line_strip, max_vertices=6) out;

uniform	mat4 PVM;
uniform mat4 VM;

uniform vec3 camDir;

// Explore culling option

 void main()
{

	vec3 ps[6];
	for (int i = 0; i < 6; ++i)
		ps[i] = vec3(VM * gl_in[i].gl_Position);
/*
	vec3 edge1 = vec3(gl_in[1].gl_Position - gl_in[0].gl_Position);
	vec3 edge2 = vec3(gl_in[2].gl_Position - gl_in[0].gl_Position);
	vec3 edge3 = vec3(gl_in[4].gl_Position - gl_in[0].gl_Position);
	vec3 edge4 = vec3(gl_in[5].gl_Position - gl_in[0].gl_Position);
	vec3 edge5 = vec3(gl_in[4].gl_Position - gl_in[2].gl_Position);
	vec3 edge6 = vec3(gl_in[3].gl_Position - gl_in[2].gl_Position);
*/
	vec3 edge1 = ps[1] - ps[0];
	vec3 edge2 = ps[2] - ps[0];
	vec3 edge3 = ps[4] - ps[0];
	vec3 edge4 = ps[5] - ps[0];
	vec3 edge5 = ps[4] - ps[2];
	vec3 edge6 = ps[3] - ps[2];


	vec3 n = normalize(cross(edge1, edge2));
	vec3 n2 = normalize(cross(edge2, edge3));
	vec3 n4 = normalize(cross(edge3, edge4));
	vec3 n6 = normalize(cross(edge6, edge5));
//	n = normalize(normalMatrix * n2);

	if (dot(n2, ps[0]) < 0)
		return;
		
	vec4 p[3];
	p[0] = PVM * gl_in[0].gl_Position;
	p[1] = PVM * gl_in[2].gl_Position;
	p[2] = PVM * gl_in[4].gl_Position;

	vec4 q[3];
	q[0] = PVM * gl_in[1].gl_Position;
	q[1] = PVM * gl_in[3].gl_Position;
	q[2] = PVM * gl_in[5].gl_Position;

	float crease = 0.1;
 	if (
	dot(n2,n) < crease 
	|| gl_in[0].gl_Position == gl_in[1].gl_Position
	|| 
	dot(n, ps[0]) < 0
	) 
	{
			gl_Position = p[0];
			EmitVertex();
 
			gl_Position = p[1];
			EmitVertex();

			EndPrimitive();
	}
	if (
	dot(n2,n4) < crease 
	|| gl_in[4].gl_Position == gl_in[5].gl_Position 
	|| 
	dot(n4, ps[0]) < 0
	) 
	{
			gl_Position = p[0];
			EmitVertex();
 
			gl_Position = p[2];
			EmitVertex();

			EndPrimitive();
	}
	if (
	dot(n2,n6) < crease 
	|| gl_in[2].gl_Position == gl_in[3].gl_Position
	|| 
	dot(n6, ps[0]) < 0
	) 
	{
			gl_Position = p[1];
			EmitVertex();
 
			gl_Position = p[2];
			EmitVertex();

			EndPrimitive();
	}
 
/*  	if (dot(cross(vec3(p[1] - p[0]), vec3(p[2] - p[0])), vec3(p[0])) > 0.0) 
	{
		// copy attributes
		if (
			gl_in[0].gl_Position == gl_in[1].gl_Position 
			|| 
			dot(cross(vec3(q[0] - p[0]), vec3(p[1] - p[0])), vec3(p[0])) < 0.0 
			||
			dot(n,n2) < 0.5
			) 
		{
			gl_Position = p[0];
			EmitVertex();
 
			gl_Position = p[1];
			EmitVertex();

			EndPrimitive();
		}

		if (
			gl_in[2].gl_Position == gl_in[3].gl_Position 
			|| 
			dot(cross(vec3(q[1] - p[1]), vec3(p[2] - p[1])), vec3(p[1])) < 0.0 
			||
			dot(n2,n6) < 0.5
			) 
		{
			gl_Position = p[1];
			EmitVertex();

			gl_Position = p[2];
			EmitVertex();

			EndPrimitive();
		}

		if (
			gl_in[4].gl_Position == gl_in[5].gl_Position 
			||
			dot(cross(vec3(p[2] - p[0]), vec3(q[2] - p[0])), vec3(p[2])) < 0.0 
			||
			dot(n2,n4) < 0.5
			)
		{
			gl_Position = p[2];
			EmitVertex();

			gl_Position = p[0];
			EmitVertex();

			EndPrimitive();
		}
	}
 */
 }
