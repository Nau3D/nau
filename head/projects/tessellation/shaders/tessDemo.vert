#version 410

uniform mat4 viewMatrix, invViewMatrix, projMatrix, invProjMatrix;

in vec4 position;
out vec4 posV;

void main() {

    posV = (viewMatrix * invViewMatrix) * (projMatrix * invProjMatrix) * position;
}



