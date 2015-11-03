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

			static std::string GetName(const std::string &fn);
			static std::string GetExtension(const std::string &fn);
			static std::string GetPath(const std::string &fn);
			static std::string GetRelativePathTo(const std::string &currentDir, const std::string &absFileName);
			static std::string GetFullPath(const std::string &currentDir, const std::string &relFileName);
			static bool IsRelative(const std::string &fn);
			static std::string CleanFullPath(const std::string &fn);
			static bool Exists(const std::string &fn);
			static std::string Validate(std::string s);
			static std::string BuildFullFileName(std::string path, std::string filename);
			static void RecurseDirectory(std::string path, std::vector<std::string> *res);
			static std::string GetCurrentFolder();
			static std::string GetAppFolder();
			static bool CreateDir(std::string path);

			static std::string TextRead(const std::string &fn);
			static int TextWrite(const std::string &fn, const std::string &s);

			static void GetFilesInFolder(std::string path, std::string extension, std::vector<std::string>* files);

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
