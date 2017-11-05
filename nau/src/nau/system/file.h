#ifndef FILE_H
#define FILE_H

#include <vector>
#include <string>

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
				FBX,
				UNKNOWN
			} FileType;

			static const std::string PATH_SEPARATOR;
			
		public:
			nau_API File(std::string filepath, bool native = false);
			
			nau_API std::string getURI (void);
			nau_API std::string getFilename (void);
			nau_API std::string getExtension (void);
			nau_API std::string getFullPath (void);
			nau_API std::string getPath (void);
			nau_API std::string getDrive (void);


			nau_API std::string getRelativeTo (std::string path);
			nau_API std::string getRelativeTo (File &file);

			nau_API std::vector<std::string>& getPathComponents (void);

			nau_API void appendToPath (std::string path);

			nau_API void setFilename (std::string aFilename);

			nau_API nau::system::File::FileType getType (void);

			/// \brief Return an absolute version of this file
			nau_API std::string getAbsolutePath ();

			nau_API ~File(void);

			static nau_API std::string GetName(const std::string &fn);
			static nau_API std::string GetNameWithoutExtension(const std::string &fn);
			static nau_API std::string GetExtension(const std::string &fn);
			static nau_API std::string GetPath(const std::string &fn);
			static nau_API std::string GetRelativePathTo(const std::string &currentDir, const std::string &absFilename);
			static nau_API std::string GetFullPath(const std::string &currentDir, const std::string &relFilename);
			static nau_API bool IsRelative(const std::string &fn);
			static nau_API std::string CleanFullPath(const std::string &fn);
			static nau_API bool Exists(const std::string &fn);
			static nau_API std::string Validate(const std::string &s);
			static nau_API std::string BuildFullFilename(std::string path, std::string filename);
			static nau_API void RecurseDirectory(std::string path, std::vector<std::string> *res);
			static nau_API std::string GetCurrentFolder();
			static nau_API std::string GetAppFolder();
			static nau_API bool CreateDir(std::string path);
			static nau_API void FixSlashes(std::string &name);

			static nau_API std::string TextRead(const std::string &fn);
			static nau_API int TextWrite(const std::string &fn, const std::string &s);

			static nau_API void GetFilesInFolder(std::string path, std::string extension, std::vector<std::string>* files);

		private:
			void construct (std::string filepath);
			void checkURI (std::string &filepath);
			std::string join (std::string head, std::string separator, bool file);
			void decode (std::string &filepath);
			std::string getCurrentWorkingDir();
			std::vector<std::string> m_vPath;
			std::string m_Filename;
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
