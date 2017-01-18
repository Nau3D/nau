#version 430

 struct ray {
	vec4 pos;
	vec4 dir;
};


layout(std430, binding = 3) buffer testBuffer {			//vec2 com posição no ecra, e um int = indice do RayBuffer
	vec2 screenPos[]; 
};

layout(std430, binding = 4) buffer RayBuffer {
	ray rays[];
}; 


layout(binding=1, offset=240) uniform atomic_uint at0;		//Light Pixels no SM Encolhido
layout(binding=1, offset=244) uniform atomic_uint at1;		//Shadow Pixels no SM Encolhido
layout(binding=1, offset=248) uniform atomic_uint at2;		//Light Pixels no SM Expandido
layout(binding=1, offset=252) uniform atomic_uint at3;		//Shadow Pixels no SM Expandido  
layout(binding=1, offset=256) uniform atomic_uint at4;		//Light Pixels ambos os SM 
layout(binding=1, offset=260) uniform atomic_uint at5;		//Shadow Pixels ambos os SM
layout(binding=1, offset=264) uniform atomic_uint at6;		//Doubt Pixels (Light no Encolhido e Shadow no Expandido)
layout(binding=1, offset=268) uniform atomic_uint at7;		//Doubt Pixels (Shadow no Encolhido e Light no Expandido)  
layout(binding=1, offset=272) uniform atomic_uint at8;		//Non Light Facing

layout(binding=1, offset=276) uniform atomic_uint at22;		//Doubt Pixels(Total)(Apontador)

uniform sampler2D texUnit, texPos;
uniform vec3 lightDirection;
uniform float tMax;

in vec2 texCoordV;

out vec4 outColor;

void main() {
//Combine	
	vec4 colorSM = texture(texUnit, texCoordV);
	bool green = (colorSM.r == 0.0 && colorSM.b == 0.0 && colorSM.g == 1.0);
	bool duvida = false;
	uint p=0;
	vec3 posicao;
	ray r;
	
	if (green) {
		atomicCounterIncrement(at8); //(PVCL)
	}
	else {
		if (colorSM.r > 0.0) { //Está em duvida
			duvida = true;
			
			if (colorSM.g > 0.0) {// Shadow/Light
				atomicCounterIncrement(at7); //Doubt Pixels (Shadow no Encolhido e Light no Expandido)
				atomicCounterIncrement(at1); //Shadow Pixels no SM Encolhido
				atomicCounterIncrement(at2); //Light Pixels no SM Expandido
			}
			else {	// Light/Shadow
				atomicCounterIncrement(at6); //Doubt Pixels (Light no Encolhido e Shadow no Expandido)
				atomicCounterIncrement(at0); //Light Pixels no SM Encolhido
				atomicCounterIncrement(at3); //Shadow Pixels no SM Expandido 
			}
		}
		else {
			if (colorSM.b > 0.0) {
					atomicCounterIncrement(at4); //Light Pixels ambos os SM 
					atomicCounterIncrement(at0); //Light Pixels no SM Encolhido
					atomicCounterIncrement(at2); //Light Pixels no SM Expandido
				}
			else {
				if (colorSM.g ==0.0 && colorSM.r==0.0 && colorSM.b==0.0) {
					atomicCounterIncrement(at5); //Shadow Pixels ambos os SM
					atomicCounterIncrement(at1); //Shadow Pixels no SM Encolhido
					atomicCounterIncrement(at3); //Shadow Pixels no SM Expandido 
				}
			}
		}
	}
	outColor = colorSM; 
	
 	if (duvida) {//criar raio e  posição de ecra (Texcoord) aqui
		//criar a posição de ecra
 		p = atomicCounterIncrement(at22); //Doubt Pixels(Total)(Apontador) - Isto vai indicar o Nº de pixeis em duvida na imagem
		screenPos[p] = texCoordV;	//multiplicar isto depois pelo tamanho da Imagen (ie x*1024, y*1024)
		
		//criar raio
		posicao = texture(texPos, texCoordV).xyz;
		r.pos = vec4(posicao, 0.01);
		r.dir = vec4(-lightDirection, tMax);
		rays[p] = r; 
	} 	
}