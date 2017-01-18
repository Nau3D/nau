#version 430

out float depth;

in DataG2F {
	vec4 pos;
	vec2 bBoxMin;
	vec2 bBoxMax;
} DataIn;

//uniform int shadowMapWidth;
//uniform int shadowMapHeight;

//testar profundidade?

void main()
{
	vec2 coord = vec2(DataIn.pos);
	
 	if (gl_Layer == 0 ) {
		//depth = 0.0;
		depth = length(DataIn.pos)/100.0;
	}
	else { //*/

		if (all(greaterThanEqual(coord, /*vec2(-1,-1)*/DataIn.bBoxMin)) && all(lessThanEqual(coord, /*vec2(1,1)*/DataIn.bBoxMax)))
			depth = length(DataIn.pos)/100.0;
		else
			discard;
	}
   

   //depth = 0.0; 

} 
