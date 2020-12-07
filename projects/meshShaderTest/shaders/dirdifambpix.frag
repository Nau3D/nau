#version 150

uniform	vec4 diffuse;
uniform vec4 l_dir;
uniform mat4 m_view;

in PerVertexData{
	vec3 normal;
};





out vec4 outputF;

void main()
{

    vec3 light_dir = normalize(vec3(m_view * -l_dir));
	float intensity;
	vec3 l, n;
	
	n = normalize(normal);
	// no need to normalize the light direction!
	intensity = max(dot(light_dir,n),0.0);
	
	outputF = max(diffuse * 0.25, diffuse * intensity);
//    outputF = vec4(light_dir,0);
    //outputF = vec4(light_dir,0);

}