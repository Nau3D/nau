#version 410

//layout(quads, equal_spacing, ccw) in;
layout(quads, fractional_odd_spacing, ccw) in;

uniform	mat4 projModelViewMatrix;
uniform	mat3 normalMatrix;



in vec4 posTC[];
out vec3 normalTE;
out vec2 texCoordTE;

void main() {

	vec4 pa = posTC[0] * (1 - gl_TessCoord.x) + posTC[3] * gl_TessCoord.x;
	
	vec4 pb = posTC[12] * (1 - gl_TessCoord.x) + posTC[15] * gl_TessCoord.x;
	
	vec4 p = pa * (1 - gl_TessCoord.y) + pb * gl_TessCoord.y;
	//gl_Position = projMatrix * modelviewMatrix * p;
	
vec4 p00 = posTC[ 0];
vec4 p10 = posTC[ 1];
vec4 p20 = posTC[ 2];
vec4 p30 = posTC[ 3];
vec4 p01 = posTC[ 4];
vec4 p11 = posTC[ 5];
vec4 p21 = posTC[ 6];
vec4 p31 = posTC[ 7];
vec4 p02 = posTC[ 8];
vec4 p12 = posTC[ 9];
vec4 p22 = posTC[10];
vec4 p32 = posTC[11];
vec4 p03 = posTC[12];
vec4 p13 = posTC[13];
vec4 p23 = posTC[14];
vec4 p33 = posTC[15];
float u = gl_TessCoord.x;
float v = gl_TessCoord.y;

texCoordTE = vec2(u,v);

// the basis functions:
float bu0 = (1.-u) * (1.-u) * (1.-u);
float bu1 = 3. * u * (1.-u) * (1.-u);
float bu2 = 3. * u * u * (1.-u);
float bu3 = u * u * u;
float dbu0 = -3. * (1.-u) * (1.-u);
float dbu1 =  3. * (1.-u) * (1.-3.*u);
float dbu2 =  3. * u *      (2.-3.*u);
float dbu3 =  3. * u *      u;
float bv0 = (1.-v) * (1.-v) * (1.-v);
float bv1 = 3. * v * (1.-v) * (1.-v);
float bv2 = 3. * v * v * (1.-v);
float bv3 = v * v * v;
float dbv0 = -3. * (1.-v) * (1.-v);
float dbv1 =  3. * (1.-v) * (1.-3.*v);
float dbv2 =  3. * v *      (2.-3.*v);
float dbv3 =  3. * v *      v;

vec4 dpdu = dbu0 * ( bv0*p00 + bv1*p01 + bv2*p02 + bv3*p03 )
+ dbu1 * ( bv0*p10 + bv1*p11 + bv2*p12 + bv3*p13 )
+ dbu2 * ( bv0*p20 + bv1*p21 + bv2*p22 + bv3*p23 )
+ dbu3 * ( bv0*p30 + bv1*p31 + bv2*p32 + bv3*p33 );
vec4 dpdv = bu0 * ( dbv0*p00 + dbv1*p01 + dbv2*p02 + dbv3*p03 )
+ bu1 * ( dbv0*p10 + dbv1*p11 + dbv2*p12 + dbv3*p13 )
+ bu2 * ( dbv0*p20 + dbv1*p21 + dbv2*p22 + dbv3*p23 )
+ bu3 * ( dbv0*p30 + dbv1*p31 + dbv2*p32 + dbv3*p33 );

normalTE = normalize(normalMatrix * cross( dpdu.xyz, dpdv.xyz ));
normalTE= vec3(normalTE.x, normalTE.z, -normalTE.y);
//normalTE = normalize(vec3(modelviewMatrix * vec4(cross( dpdu.xyz, dpdv.xyz ),0.0)));

vec4 res = bu0 * ( bv0*p00 + bv1*p01 + bv2*p02 + bv3*p03 )
+ bu1 * ( bv0*p10 + bv1*p11 + bv2*p12 + bv3*p13 )
+ bu2 * ( bv0*p20 + bv1*p21 + bv2*p22 + bv3*p23 )
+ bu3 * ( bv0*p30 + bv1*p31 + bv2*p32 + bv3*p33 );
	
vec4 res2 = vec4(res.x, res.z, -res.y, 1.0);
gl_Position = projModelViewMatrix * res2;
}

