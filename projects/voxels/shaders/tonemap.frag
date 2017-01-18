#version 430

in vec2 texPos;

uniform sampler2D origTex, lumiTex;
uniform float alpha = 0.45;

out vec4 outColor;

/// <summary>
/// Convert an sRGB pixel into a CIE xyY (xy = chroma, Y = luminance).
/// <summary>
vec3 RGB2xyY (vec3 rgb)
{
	const mat3 RGB2XYZ = mat3(0.4124, 0.3576, 0.1805,
							  0.2126, 0.7152, 0.0722,
							  0.0193, 0.1192, 0.9505);
	vec3 XYZ = RGB2XYZ * rgb;
	
	// XYZ to xyY
	return vec3(XYZ.x / (XYZ.x + XYZ.y + XYZ.z),
				XYZ.y / (XYZ.x + XYZ.y + XYZ.z),
				XYZ.y);
}


/// <summary>
/// Convert a CIE xyY value into sRGB.
/// <summary>
vec3 xyY2RGB (vec3 xyY)
{
	// xyY to XYZ
	vec3 XYZ = vec3((xyY.z / xyY.y) * xyY.x,
					xyY.z,
					(xyY.z / xyY.y) * (1.0 - xyY.x - xyY.y));

	const mat3 XYZ2RGB = mat3(3.2406, -1.5372, -0.4986,
                              -0.9689, 1.8758, 0.0415, 
                              0.0557, -0.2040, 1.0570);
	
	return XYZ2RGB * XYZ;
}

void main() {
	
	ivec2 imageCoords = ivec2(texPos * vec2(1024, 1024));
	vec2 lumi = texelFetch(lumiTex, imageCoords, 0).rg;
	vec2 lumiAccum = texelFetch(lumiTex, ivec2(0), 10).rg;
//	vec2 lumiAccum = texelFetch(lumiTex, imageCoords/16, 4).rg;
	
	vec3 c2 = texture(origTex, texPos).rgb ;

	float l = (alpha/lumiAccum.r) * lumi.r;
	
	l = (l * (1 + l/(lumiAccum.g)))/ (1 + l);
	
	vec3 xyY = RGB2xyY(c2);
	xyY.z *= l;
	c2 = xyY2RGB(xyY);
	c2 = pow(c2, vec3(1/2.2));
	outColor = vec4(c2, 1.0) ;
	//outColor = vec4(l);
}	