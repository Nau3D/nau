#version 430


layout(binding=1, offset=180) uniform atomic_uint at2;		//Light Pixels no SM Expandido
layout(binding=1, offset=184) uniform atomic_uint at3;		//Shadow Pixels no SM Expandido  
layout(binding=1, offset=188) uniform atomic_uint at8;		//White Pixels

layout(binding=1, offset=192) uniform atomic_uint at9;   	//Light Optix
layout(binding=1, offset=196) uniform atomic_uint at10;  	//Shadow Optix
layout(binding=1, offset=200) uniform atomic_uint at11;		//Background Pixels Optix

layout(binding=1, offset=204) uniform atomic_uint at23;  	//Correctos em Luz no Expandido/Prime
layout(binding=1, offset=208) uniform atomic_uint at24;		//Incorrectos em Luz no Expandido/Prime


layout(std430, binding = 2) buffer hitsBuffer {
	vec4 hits[];
}; 

uniform sampler2D texUnit; 
uniform int RenderTargetX;
uniform int RenderTargetY;

in vec3 texCoordV;

out vec4 outColor;

void main() {
 
	ivec2 coord = ivec2(texCoordV.xy*vec2(RenderTargetX,RenderTargetY));
	int coordB = coord.x* RenderTargetY + coord.y;
	vec4 resPrime = hits[coordB];
	bool intersectou = (resPrime.y >= 0.0); //y = indice do triangulo intersectado
 
//Combine	
	vec4 colorSM = texture(texUnit, texCoordV.xy);
	bool white = (colorSM.r == 1.0 && colorSM.b == 1.0 && colorSM.g == 1.0);
	outColor = colorSM; 
	
  	if (white) {
		atomicCounterIncrement(at8); //White Pixels
		atomicCounterIncrement(at11); //Background Pixels Optix
	}
	else {
		if (colorSM.b > 0.0) {
			atomicCounterIncrement(at2); //Light Pixels no SM Expandido
		}
		else {
			atomicCounterIncrement(at3); //Shadow Pixels no SM Expandido 
		}
		
		if (intersectou) {
			atomicCounterIncrement(at10); //Shadow Optix  
		}
		else {
			atomicCounterIncrement(at9); //Light Optix
		}
		
		if (colorSM.b > 0.0) {//Está em Luz no Expandido
			if (intersectou) {
				atomicCounterIncrement(at24); //Incorrectos em Luz no Expandido/Prime (Expandido:Luz / Prime: Sombra)
				outColor = vec4(1.0f,0.0f,0.0f,1.0f);
			}
			else {
				atomicCounterIncrement(at23); //Correctos em Luz no Expandido/Prime
			}
		}
		
	}
	
}