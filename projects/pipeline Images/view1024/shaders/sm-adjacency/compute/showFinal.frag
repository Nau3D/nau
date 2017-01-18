#version 430

layout(binding=1, offset=280) uniform atomic_uint atPVPL;		
layout(binding=1, offset=284) uniform atomic_uint atPVCL;		   	
layout(binding=1, offset=288) uniform atomic_uint atLight;		
layout(binding=1, offset=292) uniform atomic_uint atShadow;
layout(binding=1, offset=296) uniform atomic_uint atDoubtLight;		
layout(binding=1, offset=300) uniform atomic_uint atDoubtShadow;
layout(binding=1, offset=304) uniform atomic_uint atCorrectL;		
layout(binding=1, offset=308) uniform atomic_uint atIncorrectL;   	
layout(binding=1, offset=312) uniform atomic_uint atCorrectS;   	
layout(binding=1, offset=316) uniform atomic_uint atIncorrectS;   	
layout(binding=1, offset=320) uniform atomic_uint atTotal;  	 	
	


layout(std430, binding = 3) buffer hitsBuffer {
	vec4 hits[];
};

uniform sampler2D texUnit; //Resulatado dos ShadowMaps
uniform sampler2D texPrime; //Resultado do Compute
uniform int RenderTargetX;
uniform int RenderTargetY;

in vec3 texCoordV;

out vec4 outColor;

void main() {
 
//Combinar texUnit e texPrime	
	vec4 colorFinal = vec4(0.0);
	vec4 colorSM = texture(texUnit, texCoordV.xy);
	vec4 colorPrime = texture(texPrime, texCoordV.xy);
	bool green = (colorSM.r == 0.0 && colorSM.b == 0.0 && colorSM.g == 1.0); 
	
	if (green) { //PVCL
		colorFinal = vec4(0.0,1.0,0.0,1.0);
	}
	else {
		if (colorSM.r>0) { //Duvida
			if (colorPrime.b>0){
				colorFinal = vec4(1.0);
				atomicCounterIncrement(atDoubtLight);
			}
			else{
				colorFinal = vec4(0.0,0.0,0.0,1.0);
				atomicCounterIncrement(atDoubtShadow);
			}
		}
		else { //Não Duvida
			if (colorSM.b>0) 
				colorFinal = vec4(1.0);
			else
				colorFinal = vec4(0.0,0.0,0.0,1.0);
		}
	}
outColor = colorFinal;	

	
//Atomicos
	ivec2 coord = ivec2(texCoordV.xy*vec2(RenderTargetX,RenderTargetY));
	int coordB = coord.x* RenderTargetY + coord.y;
	vec4 resPrime = hits[coordB];
	bool intersectou = (resPrime.y >= 0.0); //y = indice do triangulo intersectado
	atomicCounterIncrement(atTotal);
	
	if (green) {
		atomicCounterIncrement(atPVCL);
	}
	else {
		atomicCounterIncrement(atPVPL);
		if(intersectou) { //Optix Prime (Puro) diz que esta em Sombra
			if (colorFinal.r==0 && colorFinal.g==0 && colorFinal.b==0){//Esta em sombra
				atomicCounterIncrement(atShadow);
				atomicCounterIncrement(atCorrectS);
			}
			else { //Esta em Luz
				atomicCounterIncrement(atLight);
				atomicCounterIncrement(atIncorrectL);
				outColor = vec4(0.0,0.0,1.0,1.0);
			}
		}
		else {
			if (colorFinal.r>0 && colorFinal.g>0 && colorFinal.b>0){//Esta em Luz
				atomicCounterIncrement(atLight);
				atomicCounterIncrement(atCorrectL);
			}
			else { //esta em sombra
				atomicCounterIncrement(atShadow);
				atomicCounterIncrement(atIncorrectS);
				outColor = vec4(1.0,0.0,0.0,1.0);
			}
		}
	}

//fim	
}