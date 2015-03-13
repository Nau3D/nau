#ifndef FILE_H
#define FILE_H

#include <vector>
#include <string>

namespace nau
{
	namespace system 
	{
		class File
		{

		public:
			typedef enum { 
				COLLADA,
				THREEDS,
				NAUBINARYOBJECT,
				WAVEFRONTOBJ,
				OGREXMLMESH,
				PATCH,
				BLENDER,
				PLY,
				LIGHTWAVE,
				STL,
				TRUESPACE,
				UNKNOWN
			} FileType;

			static const std::string PATH_SEPARATOR;

		public:
			File(std::string filepath, bool native = false);
			
			std::string getURI (void);
			std::string getFilename (void);
			std::string getExtension (void);
			std::string getFullPath (void);
			std::string getPath (void);
			std::string getDrive (void);


			std::string getRelativeTo (std::string path);
			std::string getRelativeTo (File &file);

			std::vector<std::string>& getPathComponents (void);

			void appendToPath (std::string path);

			void setFilename (std::string aFilename);

			nau::system::File::FileType getType (void);

			/// \brief Return an absolute version of this file
			std::string getAbsolutePath ();

			~File(void);

		private:
			void construct (std::string filepath);
			void checkURI (std::string &filepath);
			std::string join (std::string head, std::string separator, bool file);
			void decode (std::string &filepath);
			std::string getCurrentWorkingDir();
			std::vector<std::string> m_vPath;
			std::string m_FileName;
			std::string m_FileExtension;
			std::string m_Drive;
			bool m_IsRelative;
			bool m_IsLeaf;
			bool m_HasExtension;
			bool m_IsNative;
		};
	};
};

#endif
