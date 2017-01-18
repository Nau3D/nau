#version 430

layout(binding=1, offset=24) uniform atomic_uint atSM1;  //Light-Facing Pixels (SM)
layout(binding=1, offset=28) uniform atomic_uint atSM2;  //Not Light-Facing Pixels (SM)
layout(binding=1, offset=32) uniform atomic_uint atSM3;  //Pixels in Light (SM)
layout(binding=1, offset=36) uniform atomic_uint atSM4;  //Pixels in Shadow (SM)
layout(binding=1, offset=40) uniform atomic_uint atSM5;	 //Total Pixels (SM)

layout(binding=1, offset=44) uniform atomic_uint atC1;  //Correct Pixels in Light
layout(binding=1, offset=48) uniform atomic_uint atC2;  //Incorrect Pixels in Light
layout(binding=1, offset=52) uniform atomic_uint atC3;  //Correct Pixels in Shadow
layout(binding=1, offset=56) uniform atomic_uint atC4;  //Incorrect Pixels in Shadow




layout(std430, binding = 2) buffer hitsBuffer {
	vec4 hits[];
};

uniform sampler2D shadowMaps;
uniform int RenderTargetX;
uniform int RenderTargetY;

//uniform sampler2DArrayShadow shadowMap;

in vec2 TexCoord;

out vec4 outColor;

void main() {
	atomicCounterIncrement(atSM5); //Total
	vec4 color = vec4(0.0);
	
	ivec2 coord = ivec2(TexCoord*vec2(RenderTargetX,RenderTargetY));
	int coordB = coord.x* RenderTargetY + coord.y;
	
	vec4 colorPrime = hits[coordB];
	bool intersectou = colorPrime.y >= 0.0;
	
	vec4 colorSM = texture(shadowMaps, TexCoord);
	bool emLuz = (colorSM.r==0.0 && colorSM.g==0.0 && colorSM.b>0.0);
	bool sombra = (colorSM.r==0.0 && colorSM.g==0.0 && colorSM.b==0.0);
	bool hidden = (colorSM.r==0.0 && colorSM.g>0.0 && colorSM.b==0.0);
	
	if(colorSM.r == 1.0 && colorSM.g == 1.0 && colorSM.b == 1.0) { 
			//Background,doesn't count (Grey Pixels)
			color = vec4(0.5);
	}
	else {		
		if (!hidden) { // Light Facing Pixels
			atomicCounterIncrement(atSM1); //Light-Facing Pixels SM
		
			if (sombra) { //ShadowMap diz que esta em Sombra
				atomicCounterIncrement(atSM4); //Shadow Pixels
				
				if (!intersectou) { //Optix diz que está em Luz
					color = vec4(1,0,0,1); 
					//color = vec4(0.0f);
					atomicCounterIncrement(atC4); //Incorrect Pixels in Shadow (Red)
				}
				else { //Optix diz que está em Sombra
					//color = vec4(1,0,0,1);
					color = vec4(0.0f);
					atomicCounterIncrement(atC3); //Correct Pixels in Shadow 
				}
			}
			else {
				if (emLuz) { //ShadowMap diz que está em Luz 
					atomicCounterIncrement(atSM3); //Light Pixels
					
					if (!intersectou) { //Optix diz que está em Luz
						//color = vec4(1,1,0,1);
						color = vec4(1.0f);
						atomicCounterIncrement(atC1); //Correct Pixels in Light
					}
					else { //Optix diz que está em Sombra
						color = vec4(0,0,1,1);
						//color = vec4(1.0);
						atomicCounterIncrement(atC2); //Incorrect Pixels in Light (Blue)
					}
				}
				else { //ShadowMap diz que esta em Duvida
					color = vec4(1,0,1,1); 
				}
			}
		}
		else { //not Light Facing Pixels
			color = vec4(0,1,0,1);
			atomicCounterIncrement(atSM2); //Not Light-Facing Pixels (SM)
		}
	}

	outColor = color; 
	
}
