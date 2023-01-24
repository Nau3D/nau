#version 460
 
out vec4 colorOut;
 
in PerVertexData
{
  vec4 color;
} fragIn;  
 
void main()
{
  colorOut = fragIn.color;
}