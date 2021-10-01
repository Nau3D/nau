#ifndef BUFFERLOADER_H
#define BUFFERLOADER_H

#include "nau/material/iBuffer.h"

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

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
			static int LoadBuffer (nau::material::IBuffer *aBuffer, std::string &aFilename);
			static nau_API int SaveBuffer(nau::material::IBuffer *aBuffer, bool forceBinary = false);

		};
	};
};

#endif 