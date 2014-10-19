#include "NauGlBufferInfo.h"


NauGlBufferInfo::NauGlBufferInfo() :
index(0),
size(0),
components(0),
type(0),
stride(0),
normalized(0),
divisor(0),
integer(0),
isVAO(false)
{}


NauGlBufferInfo::NauGlBufferInfo(int indexp, int sizep) :
index(indexp),
size(sizep),
components(0),
type(0),
stride(0),
normalized(0),
divisor(0),
integer(0),
isVAO(false)
{
}


NauGlBufferInfo::NauGlBufferInfo(int indexp, int sizep, int componentsp, int typep, int stridep, int normalizedp, int divisorp, int integerp) :
index(indexp),
size(sizep),
components(componentsp),
type(typep),
stride(stridep),
normalized(normalizedp),
divisor(divisorp),
integer(integerp),
isVAO(true)
{
}

bool NauGlBufferInfo::isVAOBuffer(){
	return isVAO;
}