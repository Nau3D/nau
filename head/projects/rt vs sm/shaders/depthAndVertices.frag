#version 410

flat in vec4 vertexA, vertexB, vertexC;

out vec4 vA, vB, vC;

void main(void) {

	vA = vertexA;
	vB = vertexB;
	vC = vertexC;
}
