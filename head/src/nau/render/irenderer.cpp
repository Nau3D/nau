#include <nau/render/irenderer.h>

const std::string IRenderer::MatrixTypeString[] = {"PROJECTION", "MODEL", 
												"VIEW", "TEXTURE",
												"VIEW_MODEL", "PROJECTION_VIEW_MODEL", 
												"PROJECTION_VIEW", "TS05_PVM", "NORMAL"};


std::map<int, std::string> IRenderer::AtomicLabels;
int IRenderer::AtomicLabelsCount = 0;


void 
IRenderer::addAtomic(int id, std::string name) {

	AtomicLabelsCount++;
	AtomicLabels[id] = name;
}


void 
IRenderer::getPropId(std::string &s, int *id){

	// value returned in case of an invalid string
	*id = -1;

	for (int i = 0; i < COUNT_MATRIXTYPE; i++) {

		if (s == MatrixTypeString[i]) {
		
			*id = i;
			return;
		}
	}
}


const std::string &
IRenderer::getPropMatrixTypeString(MatrixType mode) 
{
	return MatrixTypeString[mode];
}


int 
IRenderer::translateStringToStencilOp(std::string s) {

	if (s == "KEEP")
		return(KEEP);
	if (s == "ZERO")
		return(ZERO);
	if (s == "REPLACE")
		return(REPLACE);
	if (s == "INCR")
		return(INCR);
	if (s == "INCR_WRAP")
		return(INCR_WRAP);
	if (s == "DECR")
		return(DECR);
	if (s == "DECR_WRAP")
		return(DECR_WRAP);
	if (s == "INVERT")
		return(INVERT);

	return(-1);
}

int 
IRenderer::translateStringToStencilFunc(std::string s) {

	if (s == "NEVER")
		return(NEVER);
	if (s == "ALWAYS")
		return(ALWAYS);
	if (s == "LESS")
		return(LESS);
	if (s == "LEQUAL")
		return(LEQUAL);
	if (s == "GEQUAL")
		return(GEQUAL);
	if (s == "GREATER")
		return(GREATER);
	if (s == "EQUAL")
		return(EQUAL);
	if (s == "NOT_EQUAL")
		return(NOT_EQUAL);

	return(-1);
}