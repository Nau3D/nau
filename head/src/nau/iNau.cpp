#include "iNau.h"

INau *INau::Interface;

INau 
*INau::GetInterface() {

	return Interface;
}


void 
INau::SetInterface(INau *n) {

	Interface = n;
}
