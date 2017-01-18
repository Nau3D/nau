#version 430

uniform mat4 PVM, V, M;
uniform mat3 NormalMatrix;
uniform vec4 lightDirection;
uniform mat4 lightSpaceMat;
uniform float GridSizeF;

layout(triangles_adjacency) in; //recebo a forma do GL_TRIANGLES ADJACENCY (6 vertices) 
layout (triangle_strip, max_vertices=100) out;

in Data {
	vec4 pos;
	vec3 normal;
	int index;
} DataIn[];
 

out DataG2F {
	vec4 pos;
	vec2 bBoxMin;
	vec2 bBoxMax;
} DataOut;

//uniform float pixelDiagonal;

//Tenta ver o que a informação da lista de adjacencia

//Tenta por o composer a render apenas um dos triangulos


vec3 lineLineIntersection(vec3 s1, vec3 e1, vec3 s2, vec3 e2, vec3 n, vec3 p) {

	vec3 result;
	
	float a1 = e1.y - s1.y;
	float b1 = s1.x - e1.x;
	float c1 = a1 * s1.x + b1 * s1.y;
	
	float a2 = e2.y - s2.y;
	float b2 = s2.x - e2.x;
	float c2 = a2 * s2.x + b2 * s2.y;
	
	float delta = a1*b2 - a2*b1;
	result = vec3(b2*c1 - b1*c2, a1*c2 - a2*c1, 0) / delta;
	
	// now result is the 2D coordinate in x,y we still need to compute z
	// Ax + By + Cz + D = 0 => z = -(Ax + By + D) / C
	if (n.z != 0) { 
		float d = -dot(n, p);
		result.z = -(n.x * result.x + n.y * result.y + d) /  n.z;
	}
	else // this would only happen with a triangle which is perpendicular to the screen
		result.z = 0;
		
	return result;

}

bool inversion(in vec3 pedge[3], in vec3 pos[3], in vec4 p[6]) {

	float c;
	float prior[3], after[3];
	vec3 line[3];
	
	for (int i = 0 ; i < 3 ; ++i) {
		line[i].x = pedge[i].x;
		line[i].y = pedge[i].y;
		line[i].z = -pedge[i].x * p[2*i].x - pedge[i].y * p[2*i].y;
	}
		
	prior[0] = p[4].x * line[0].x + p[4].y * line[0].y + line[0].z;
	prior[1] = p[0].x * line[1].x + p[0].y * line[1].y + line[1].z;
	prior[2] = p[2].x * line[2].x + p[2].y * line[2].y + line[2].z;
	
	after[0] = pos[2].x * line[0].x + pos[2].y * line[0].y + line[0].z;
	after[1] = pos[0].x * line[1].x + pos[0].y * line[1].y + line[1].z;
	after[2] = pos[1].x * line[2].x + pos[1].y * line[2].y + line[2].z;
	
	if (prior[0] * after[0] <= 0 || prior[1] * after[1] <= 0 || prior[2] * after[2] <= 0)
		return true;
	else
		return false;
}

//Este é necessario: Este é igual ao Stefan (Encolher sem Adjacencia)
void shrinkTriangle2(in vec4 p[6], in vec3 n[6], float pixelDiagonal) {

	vec3 pos[3];
	vec3 lightDir = normalize (vec3(V * -lightDirection));

	vec3 edge[3], shifta[3], shiftb[3];
	vec3 pedge[3];
	vec3 normal,normalss;
	
	//triangulo central 0-2-4
	edge[0] = p[2].xyz - p[0].xyz; //vertice 0 -> vertice 2
	edge[1] = p[4].xyz - p[2].xyz; //vertice 2 -> vertice 4
	edge[2] = p[0].xyz - p[4].xyz; //vertice 4 -> vertice 0
	float kk = 2.0;
	 
	if (length(edge[0].xy) < pixelDiagonal*kk || 
		length(edge[1].xy) < pixelDiagonal*kk || 
		length(edge[2].xy) < pixelDiagonal*kk)
		    return;
		  
	// normalss = normalize(cross(vec3(edge[0].xy, 0), vec3(edge[1].xy,0)));
	 // if (normalss.z > 0)
		 // return;
	normal = normalize(cross(edge[0], edge[1]));
	if (normal.z > 0)
		  return;
	
	
	// perpendicular edges in screen space
	pedge[0] = normalize(vec3(edge[0].y, -edge[0].x, 0)); 
	pedge[1] = normalize(vec3(edge[1].y, -edge[1].x, 0));
	pedge[2] = normalize(vec3(edge[2].y, -edge[2].x, 0));
	
	
	// shifted vertices for each edge in screen space
	shifta[0] = vec3(p[0]) + pedge[0] * pixelDiagonal;
	shiftb[0] = vec3(p[2]) + pedge[0] * pixelDiagonal;
	                 
	shifta[1] = vec3(p[2]) + pedge[1] * pixelDiagonal;
	shiftb[1] = vec3(p[4]) + pedge[1] * pixelDiagonal;
	                 
	shifta[2] = vec3(p[4]) + pedge[2] * pixelDiagonal;
	shiftb[2] = vec3(p[0]) + pedge[2] * pixelDiagonal;
	
	// new vertex positions in clip space
	// pos0 is the intersection of new edge 0 with new edge 2
	pos[0] = lineLineIntersection(shifta[0], shiftb[0], shifta[2], shiftb[2], normal, vec3(p[0]));
	// pos1 is the intersection of new edge 0 with new edge 1
	pos[1] = lineLineIntersection(shifta[0], shiftb[0], shifta[1], shiftb[1], normal, vec3(p[0]));
	// pos2 is the intersection of new edge 1 with new edge 2
	pos[2] = lineLineIntersection(shifta[2], shiftb[2], shifta[1], shiftb[1], normal, vec3(p[0]));
	
	vec3 newEdge0 = pos[1] - pos[0];
	vec3 newEdge1 = pos[2] - pos[0];
	vec3 newEdge2 = pos[2] - pos[1];
	
	// testar inversões
	if (inversion(pedge, pos, p))
		  return;
	
	float k = 2.0;
	if (length(newEdge0.xy) < pixelDiagonal*k || 
		length(newEdge1.xy) < pixelDiagonal*k || 
		length(newEdge2.xy) < pixelDiagonal*k)
		return;
		
	vec3 nn = cross(vec3(newEdge0.xy, 0), vec3(newEdge1.xy, 0));
	//if (nn.z >= -0.0)
	//	 return;
	
// Emitir vertices
	gl_Layer = 0;
	DataOut.pos = vec4(pos[0], 1);
	DataOut.bBoxMin = vec2(0);
	DataOut.bBoxMax = vec2(0);
	
	gl_Position = vec4(pos[0], 1);
	//gl_Position = p[0];
	EmitVertex();

	gl_Position = vec4(pos[1], 1);
	//gl_Position = p[2];
	EmitVertex();

	gl_Position = vec4(pos[2], 1);
	//gl_Position = p[4];
	EmitVertex();

	EndPrimitive();

}

//Corrigir este
void shrinkTriangle3(in vec4 p[6], in vec3 n[6], float pixelDiagonal) {
	
	vec3 pos[3];
	vec3 lightDir = normalize (vec3(V * -lightDirection));

	vec3 edge[3], shifta[3], shiftb[3];
	vec3 pedge[3];
	vec3 normal,normalss;
		
	//triangulo central 0-2-4
	edge[0] = p[2].xyz - p[0].xyz; //vertice 0 -> vertice 2
	edge[1] = p[4].xyz - p[2].xyz; //vertice 2 -> vertice 4
	edge[2] = p[0].xyz - p[4].xyz; //vertice 4 -> vertice 0
	
	float kk = 2.0;
	if (length(edge[0].xy) < pixelDiagonal*kk || 
		 length(edge[1].xy) < pixelDiagonal*kk || 
		 length(edge[2].xy) < pixelDiagonal*kk)
		     return;
		  
/* 	normalss = normalize(cross(vec3(edge[0].xy, 0), vec3(edge[1].xy,0)));
	 if (normalss.z > 0)
		return; */
	normal = normalize(cross(edge[0], edge[1]));
	if (normal.z > 0)
		  return;
	
	bool t1Luz, t2Luz, t3Luz; 
	//Indicadores se os triangulos adjacentes existem/estão em luz
	
	vec3 edgeA_T1, edgeA_T2, edgeA_T3;
	vec3 edgeB_T1, edgeB_T2, edgeB_T3;
	
	edgeA_T1 = p[1].xyz - p[0].xyz;
	edgeB_T1 = p[2].xyz - p[1].xyz;
	edgeA_T2 = p[3].xyz - p[2].xyz;
	edgeB_T2 = p[4].xyz - p[3].xyz;
	edgeA_T3 = p[5].xyz - p[4].xyz;
	edgeB_T3 = p[0].xyz - p[5].xyz;
	
  	if ( !(length(edgeA_T1) < 0.001 )){ //p[1] existe
		t1Luz = false;
		
		//Ver se ele está em luz ver se triangulo adjacente T1 esta em Luz
		normalss = normalize( cross(vec3(edgeA_T1), vec3(edgeB_T1) ));
		if (dot(normalss, lightDir) > 0) 
			t1Luz=true;
	}
	else { //p[1] não existe
		t1Luz = true;
	}
	
	if ( !(length(edgeA_T2) < 0.001 )){ //p[3] existe
		t2Luz = false; 
		
		//Ver se ele está em luz ver se triangulo adjacente T2 esta em Luz
		normalss = normalize( cross(vec3(edgeA_T2), vec3(edgeB_T2)));
		if (dot(normalss, lightDir) > 0) //Ver se ele está em luz
			t2Luz = true;
	}
	else { //p[3] não existe
		t2Luz = true;
	}
	
	if ( !(length(edgeA_T3) < 0.001 )){ //p[5] existe
		t3Luz = false; 
		
		//Ver se ele está em luz ver se triangulo adjacente T2 esta em Luz
		normalss = normalize( cross(vec3(edgeA_T3), vec3(edgeB_T3) ));
		if (dot(normalss, lightDir) > 0) //Ver se ele está em luz
			t3Luz = true;
	} 
	else { //p[5] não existe
		t3Luz = true;
	}
	
	// perpendicular edges in screen space
	pedge[0] = normalize(vec3(edge[0].y, -edge[0].x, 0)); 
	pedge[1] = normalize(vec3(edge[1].y, -edge[1].x, 0));
	pedge[2] = normalize(vec3(edge[2].y, -edge[2].x, 0));
	
	
	// shifted vertices for each edge in screen space
	shifta[0] = vec3(p[0]) + pedge[0] * pixelDiagonal;
	shiftb[0] = vec3(p[2]) + pedge[0] * pixelDiagonal;
	                 
	shifta[1] = vec3(p[2]) + pedge[1] * pixelDiagonal;
	shiftb[1] = vec3(p[4]) + pedge[1] * pixelDiagonal;
	                 
	shifta[2] = vec3(p[4]) + pedge[2] * pixelDiagonal;
	shiftb[2] = vec3(p[0]) + pedge[2] * pixelDiagonal;
	
	
	vec3 newPos[3], t1Aux[2], t2Aux[2], t3Aux[2];
	
	// new vertex positions in clip space (Shrunken Triangle)
	newPos[0] = lineLineIntersection(shifta[0], shiftb[0], shifta[2], shiftb[2], normal, vec3(p[0])); //pos0 is the intersection of new edge 0 with new edge 2
	newPos[1] = lineLineIntersection(shifta[0], shiftb[0], shifta[1], shiftb[1], normal, vec3(p[0])); //pos1 is the intersection of new edge 0 with new edge 1
	newPos[2] = lineLineIntersection(shifta[2], shiftb[2], shifta[1], shiftb[1], normal, vec3(p[0])); //pos2 is the intersection of new edge 1 with new edge 2
	
	//Auxilar Intersections for T1
	t1Aux[0] = lineLineIntersection(vec3(p[0]), vec3(p[2]), newPos[2], newPos[0], normal, vec3(p[0])); //Intersection of old Edge0 with new Edge2 -> Pos[0]
	t1Aux[1] = lineLineIntersection(vec3(p[0]), vec3(p[2]), newPos[1], newPos[2], normal, vec3(p[0])); //Intersection of old Edge0 with new Edge1 -> Pos[1]
	
	//Auxilar Intersections for T2
	t2Aux[0] = lineLineIntersection(vec3(p[2]), vec3(p[4]), newPos[0], newPos[1], normal, vec3(p[0])); //Intersection of old Edge1 with new Edge0 -> Pos[1]
	t2Aux[1] = lineLineIntersection(vec3(p[2]), vec3(p[4]), newPos[2], newPos[0], normal, vec3(p[0])); //Intersection of old Edge1 with new Edge2 -> Pos[2]
		
	//Auxilar Intersections for T3
	t3Aux[0] = lineLineIntersection(vec3(p[4]), vec3(p[0]), newPos[0], newPos[1], normal, vec3(p[0])); //Intersection of old Edge2 with new Edge0 -> Pos[0]
	t3Aux[1] = lineLineIntersection(vec3(p[4]), vec3(p[0]), newPos[1], newPos[2], normal, vec3(p[0])); //Intersection of old Edge2 with new Edge1 -> Pos[2]
	
	vec3 newEdge0 = newPos[1] - newPos[0];
	vec3 newEdge1 = newPos[2] - newPos[0];
	vec3 newEdge2 = newPos[2] - newPos[1];
	
	vec3 newEdge4 = t1Aux[1] - t1Aux[0];
	vec3 newEdge5 = t2Aux[1] - t2Aux[0];
	vec3 newEdge6 = t3Aux[1] - t3Aux[0];	
	
	newEdge4 = newEdge0;
	newEdge5 = newEdge2;
	newEdge6 = newEdge2;
		 
	// testar inversões
	pos[0] = newPos[0];
	pos[1] = newPos[1];
	pos[2] = newPos[2];
	
 	if (inversion(pedge, pos, p))
		  return;
	
	float k = 2.0;
	if (length(newEdge0.xy) < pixelDiagonal*k || 
				  length(newEdge1.xy) < pixelDiagonal*k || 
				  length(newEdge2.xy) < pixelDiagonal*k)
		return;
	
		
	vec3 nn = cross(vec3(newEdge0.xy, 0), vec3(newEdge1.xy, 0));
	if (nn.z >= -0.0)
		 return;
	
	// Emitir vertices
	gl_Layer = 0;
	DataOut.pos = vec4(pos[0], 1);
	DataOut.bBoxMin = vec2(0);
	DataOut.bBoxMax = vec2(0);
	
	//t1Luz=true; t2Luz=true; t3Luz=true;
	
  	if (t1Luz && t2Luz && t3Luz) { //Todos os Adjacentes estão em Luz/não Existem
		gl_Position = vec4(pos[0], 1);
		EmitVertex();

		gl_Position = vec4(pos[1], 1);
		EmitVertex();

		gl_Position = vec4(pos[2], 1);
		EmitVertex();
		EndPrimitive();
	}

//Begin 1 em Sombra
 	if (!t1Luz && t2Luz && t3Luz){ //T1 está em Sombra e os outros estão em Luz	
		gl_Position = vec4(t1Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t1Aux[1], 1);
		EmitVertex();
		gl_Position = vec4(newPos[2], 1);
		EmitVertex();
		EndPrimitive();
	} 
 	if (!t2Luz && t1Luz && t3Luz){ //T2 está em Sombra e os outros estão em Luz	
		gl_Position = vec4(newPos[0], 1);
		EmitVertex();
		gl_Position = vec4(t2Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t2Aux[1], 1);
		EmitVertex();
		EndPrimitive();
	} 
	if (!t3Luz && t1Luz && t2Luz){ //T3 está em Sombra e os outros estão em Luz
		gl_Position = vec4(t3Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(newPos[1], 1);
		EmitVertex();
		gl_Position = vec4(t3Aux[1], 1);
		EmitVertex();
		EndPrimitive();
	}
//End 1 em Sombra
	
//Begin 2 em Sombra
	if (t1Luz && !t2Luz && !t3Luz){ //T2 e T3 estão em Sombra e o T1 está em Luz	
 		gl_Position = vec4(t3Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t2Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t3Aux[1], 1); 
		EmitVertex();
 		EndPrimitive();
		
		gl_Position = vec4(t2Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t2Aux[1], 1);
		EmitVertex();
		gl_Position = vec4(t3Aux[1], 1);
		EmitVertex();
		EndPrimitive();
	}
	if (t2Luz && !t1Luz && !t3Luz){ //T1 e T3 estão em Sombra e o T2 está em Luz		
		gl_Position = vec4(t1Aux[1], 1); 
		EmitVertex();
		gl_Position = vec4(t3Aux[1], 1);
		EmitVertex();
		gl_Position = vec4(t3Aux[0], 1);
		EmitVertex();
 		EndPrimitive();
		
		gl_Position = vec4(t3Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t1Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t1Aux[1], 1);
		EmitVertex();
		EndPrimitive();
	}
	if (t3Luz && !t1Luz && !t2Luz){ //T1 e T2 estão em Sombra e o T3 está em Luz
		gl_Position = vec4(t1Aux[0], 1); 
		EmitVertex();
		gl_Position = vec4(t1Aux[1], 1);
		EmitVertex();
		gl_Position = vec4(t2Aux[1], 1);
		EmitVertex();
 		EndPrimitive();
		
		gl_Position = vec4(t2Aux[1], 1);
		EmitVertex();
		gl_Position = vec4(t1Aux[1], 1);
		EmitVertex();
		gl_Position = vec4(t2Aux[0], 1);
		EmitVertex();
		EndPrimitive();
	} 
//End 2 em Sombra
	
//Begin 3 em Sombra
 	if (!t1Luz && !t2Luz && !t3Luz) { //criar um modelo para testar este caso (um conjunto de triangulos como no caderno)	
 		//NT1
		gl_Position = vec4(t3Aux[1], 1); 
		EmitVertex();
		gl_Position = vec4(t3Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t1Aux[0], 1); 
		EmitVertex();
		EndPrimitive();
		
		//NT2
		gl_Position = vec4(t1Aux[0], 1); 
		EmitVertex();
		gl_Position = vec4(t2Aux[1], 1);
		EmitVertex();
		gl_Position = vec4(t3Aux[1], 1); 
		EmitVertex();
		EndPrimitive();
				
		//NT3
		gl_Position = vec4(t2Aux[1], 1); 
		EmitVertex();
		gl_Position = vec4(t1Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t1Aux[1], 1); 
		EmitVertex();
		EndPrimitive(); 
		
		//NT4
		gl_Position = vec4(t1Aux[1], 1); 
		EmitVertex();
		gl_Position = vec4(t2Aux[0], 1);
		EmitVertex();
		gl_Position = vec4(t2Aux[1], 1); 
		EmitVertex();
		EndPrimitive();
	} 
//End 3 Em Sombra

}


void expandTriangle3(in vec4 p[6], in vec3 n[6], in float pixelDiagonal, in vec2 bMin, in vec2 bMax) {

	vec3 pos[3];
	vec3 edge[3], shifta[3], shiftb[3];
	vec3 pedge[3];
	vec3 normal,normalss;
	
	//triangulo central 0-2-4
	edge[0] = p[2].xyz - p[0].xyz; //vertice 0 -> vertice 2
	edge[1] = p[4].xyz - p[2].xyz; //vertice 2 -> vertice 4
	edge[2] = p[0].xyz - p[4].xyz; //vertice 4 -> vertice 0
	 float kk = 2.0;
	// if (length(edge[0].xy) < pixelDiagonal*kk || length(edge[1].xy) < pixelDiagonal*kk || length(edge[2].xy) < pixelDiagonal*kk)
		  // return;
		  
	normalss = normalize(cross(vec3(edge[0].xy, 0), vec3(edge[1].xy,0)));
	 if (normalss.z > 0)
		return;
	normal = normalize(cross(edge[0], edge[1]));
	if (normal.z > 0)
	 	return;
	
	// perpendicular edges in screen space
	pedge[0] = normalize(vec3(edge[0].y, -edge[0].x, 0)); 
	pedge[1] = normalize(vec3(edge[1].y, -edge[1].x, 0));
	pedge[2] = normalize(vec3(edge[2].y, -edge[2].x, 0));
	
	// shifted vertices for each edge (Assim está a expandir)
	shifta[0] = vec3(p[0]) - pedge[0] * pixelDiagonal;
	shiftb[0] = vec3(p[2]) - pedge[0] * pixelDiagonal;
	                 
	shifta[1] = vec3(p[2]) - pedge[1] * pixelDiagonal;
	shiftb[1] = vec3(p[4]) - pedge[1] * pixelDiagonal;
	                 
	shifta[2] = vec3(p[4]) - pedge[2] * pixelDiagonal;
	shiftb[2] = vec3(p[0]) - pedge[2] * pixelDiagonal;
	
	// new vertex positions
	// pos0 is the intersection of new edge 0 with new edge 2
	pos[0] = lineLineIntersection(shifta[0], shiftb[0], shifta[2], shiftb[2], normal, vec3(p[0]));
	//pos[0] = shiftb[2];
	// pos1 is the intersection of new edge 0 with new edge 1
	pos[1] = lineLineIntersection(shifta[0], shiftb[0], shifta[1], shiftb[1], normal, vec3(p[0]));
	// pos2 is the intersection of new edge 1 with new edge 2
	pos[2] = lineLineIntersection(shifta[2], shiftb[2], shifta[1], shiftb[1], normal, vec3(p[0]));
	
	vec3 newEdge0 = pos[1] - pos[0];
	vec3 newEdge1 = pos[2] - pos[0];
	vec3 newEdge2 = pos[2] - pos[1];
	
	 // if (length(newEdge0.xy) < pixelDiagonal*kk || 
				// length(newEdge1.xy) < pixelDiagonal*kk || 
				// length(newEdge2.xy) < pixelDiagonal*kk)
		 // return;
		
	vec3 nn = cross(vec3(newEdge0.xy, 0), vec3(newEdge1.xy, 0));
	if (nn.z >= -0.0)
		return;
	
// Emitir vertices
	gl_Layer = 1;
	DataOut.pos = vec4(pos[0], 1);
	DataOut.bBoxMin = bMin;
	DataOut.bBoxMax = bMax;
	gl_Position = vec4(pos[0], 1);
//	gl_Position = p[0];
	EmitVertex();

	DataOut.pos = vec4(pos[1], 1);
	DataOut.bBoxMin = bMin;
	DataOut.bBoxMax = bMax;
	gl_Position = vec4(pos[1], 1);
//	gl_Position = p[2];
	EmitVertex();

	DataOut.pos = vec4(pos[2], 1);
	DataOut.bBoxMin = bMin;
	DataOut.bBoxMax = bMax;
	gl_Position = vec4(pos[2], 1);
//	gl_Position = p[4];
	EmitVertex();

	EndPrimitive();
}


void main() {

	//1-Expansão
	float a1 = 2.0f / GridSizeF; //comprimento
	float b1 = 2.0f / GridSizeF; //largura
	float pixelDiagonal1 = sqrt(a1*a1 + b1*b1); 
	//float pixelDiagonal = sqrt(2.0f) * 1.0f / 1024.0f; 
	
	//2-Encolher
	float a2 = 1.0f / GridSizeF; //comprimento
	float b2 = 1.0f / GridSizeF; //largura
	float pixelDiagonal2 = sqrt(a2*a2 + b2*b2); 

	//Calculate screen coordinates for the triangle
	vec4 screenPos[6];
	screenPos[0] = gl_in[0].gl_Position; //vertice 0 do triangulo central	(0-2-4)
	screenPos[1] = gl_in[1].gl_Position; //vertice 1 do triangulo 1			(0-1-2) 
	screenPos[2] = gl_in[2].gl_Position; //vertice 2 do triangulo central   (0-2-4)
	screenPos[3] = gl_in[3].gl_Position; //vertice 3 do triangulo 2			(2-3-4)
	screenPos[4] = gl_in[4].gl_Position; //vertice 4 do triangulo central   (0-2-4)
	screenPos[5] = gl_in[5].gl_Position; //vertice 5 do triangulo 3			(4-5-0) 

 	screenPos[0] /= screenPos[0].w;
	screenPos[1] /= screenPos[1].w;
	screenPos[2] /= screenPos[2].w;
 	screenPos[3] /= screenPos[3].w;
	screenPos[4] /= screenPos[4].w;
	screenPos[5] /= screenPos[5].w; 

	vec3 n[6];
	n[0] = normalize(DataIn[0].normal);
	n[1] = normalize(DataIn[1].normal);
	n[2] = normalize(DataIn[2].normal);
 	n[3] = normalize(DataIn[3].normal);
	n[4] = normalize(DataIn[4].normal);
	n[5] = normalize(DataIn[5].normal);
	
	//calculate screen space bounding box to be used
	// for clipping in the fragment shader
	// Extend and set AABB.
 	vec2 bMin, bMax; 
	bMin = min(screenPos[0].xy, min(screenPos[2].xy,screenPos[4].xy)) - pixelDiagonal2;
	bMax = max(screenPos[0].xy, max(screenPos[2].xy,screenPos[4].xy)) + pixelDiagonal2;

	//Shrink Layer 0
	shrinkTriangle2(screenPos, n, pixelDiagonal2);
	//shrinkTriangle3(screenPos, n, pixelDiagonal2);
	
	//Expand Layer 1
	expandTriangle3(screenPos, n, pixelDiagonal2, bMin, bMax);


}

/* 	//Triangulo BDA (esta direito)
	if (DataIn[0].index == 2 && DataIn[2].index == 0 && DataIn[4].index == 1) {
		//return;
	}
	//Triangulo CBA (esta mal : Devia fazer 2 em sombra, Esta a fazer 1 em Sombra)
	if (DataIn[0].index == 3 && DataIn[2].index == 2 && DataIn[4].index == 1) {
		//return;
	}
	//Triangulo FBC (esta mal : Devia fazer 2 em sombra, Esta a fazer 1 em Sombra)
	if (DataIn[0].index == 4 && DataIn[2].index == 2 && DataIn[4].index == 3) {
		//return;
	}
	//Triangulo GFC (esta mal : Devia ser 1 em sombra ele ta a fazer todos em sombra)
	if (DataIn[0].index == 5 && DataIn[2].index == 4 && DataIn[4].index == 3) {
		//return;
	} */
