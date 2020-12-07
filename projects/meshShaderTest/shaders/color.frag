#version 460
 
out vec4 colorOut;
 
in PerVertexData
{
  vec4 color;
} fragIn;  
 
void main()
{
  colorOut = vec4(1,0,0,1);//fragIn.color;
}