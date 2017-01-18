#version 430

/* layout(binding=1, offset=0) uniform atomic_uint at0;		//Light Pixels no SM Encolhido
layout(binding=1, offset=4) uniform atomic_uint at1;		//Shadow Pixels no SM Encolhido
layout(binding=1, offset=8) uniform atomic_uint at2;		//Light Pixels no SM Expandido
layout(binding=1, offset=12) uniform atomic_uint at3;		//Shadow Pixels no SM Expandido  
layout(binding=1, offset=16) uniform atomic_uint at4;		//Light Pixels ambos os SM 
layout(binding=1, offset=20) uniform atomic_uint at5;		//Shadow Pixels ambos os SM
layout(binding=1, offset=24) uniform atomic_uint at6;		//Doubt Pixels (Light no Encolhido e Shadow no Expandido)
layout(binding=1, offset=28) uniform atomic_uint at7;		//Doubt Pixels (Shadow no Encolhido e Light no Expandido)  
layout(binding=1, offset=32) uniform atomic_uint at8;		//White Pixels
layout(binding=1, offset=88) uniform atomic_uint at22;		//Doubt Pixels(Total)(Apontador)
 */
uniform sampler2D texUnit; 

in vec3 texCoordV;

out vec4 outColor;

void main() {
//Combine	
	vec4 colorSM = texture(texUnit, texCoordV.xy);
	bool white = (colorSM.r == 1.0 && colorSM.b == 1.0 && colorSM.g == 1.0);
	
	if (white) {
		//atomicCounterIncrement(at8); //White Pixels
	}
	else {
		if (colorSM.r > 0.0) { //Está em duvida
			//atomicCounterIncrement(at22); //Doubt Pixels(Total)(Apontador)
			
			if (colorSM.g > 0.0) {// Shadow/Light
				//atomicCounterIncrement(at7); //Doubt Pixels (Shadow no Encolhido e Light no Expandido)
				//atomicCounterIncrement(at1); //Shadow Pixels no SM Encolhido
				//atomicCounterIncrement(at2); //Light Pixels no SM Expandido
			}
			else {	// Light/Shadow
				//atomicCounterIncrement(at6); //Doubt Pixels (Light no Encolhido e Shadow no Expandido)
				//atomicCounterIncrement(at0); //Light Pixels no SM Encolhido
				//atomicCounterIncrement(at3); //Shadow Pixels no SM Expandido 
			}
		}
		else {
			if (colorSM.b > 0.0) {
					//atomicCounterIncrement(at4); //Light Pixels ambos os SM 
					//atomicCounterIncrement(at0); //Light Pixels no SM Encolhido
					//atomicCounterIncrement(at2); //Light Pixels no SM Expandido
				}
			else {
				if (colorSM.g ==0.0 && colorSM.r==0.0 && colorSM.b==0.0) {
					//atomicCounterIncrement(at5); //Shadow Pixels ambos os SM
					//atomicCounterIncrement(at1); //Shadow Pixels no SM Encolhido
					//atomicCounterIncrement(at3); //Shadow Pixels no SM Expandido 
				}
			}
		}
	}
	outColor = colorSM; 
}