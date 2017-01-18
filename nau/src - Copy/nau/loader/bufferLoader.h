#ifndef BUFFERLOADER_H
#define BUFFERLOADER_H

#include "nau/material/iBuffer.h"

namespace nau 
{

	namespace loader 
	{
		// Loads buffer contents from file
		class BufferLoader
		{
		public:

			// Load Scene
			/// returns the number of bytes read 
			static int loadBuffer (nau::material::IBuffer *aBuffer, std::string &aFilename);                                                       
		};
	};
};

#endif 