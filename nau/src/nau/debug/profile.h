// profile.h: interface for the Profile class.
//
//////////////////////////////////////////////////////////////////////


#ifndef PROFILE_H
#define PROFILE_H

#define NAU_PROFILE_NONE 0
#define NAU_PROFILE_CPU 1
#define NAU_PROFILE_CPU_AND_GPU 2


#define NAU_PROFILE NAU_PROFILE_CPU_AND_GPU



// Available clocks
// note that gettimeofday is not available on windows!
#define PROFILE_C_CLOCK 0
#define PROFILE_GETTIMEOFDAY 1
#define PROFILE_WIN_HIGH_PERFORMANCE_COUNTER 2
#define PROFILE_WIN_SYSTEMTIME 3


// Use this define to select your favourite clock!
#define PROFILE_CLOCK PROFILE_C_CLOCK


// do the required includes for each clock
#if PROFILE_CLOCK == PROFILE_WIN_HIGH_PERFORMANCE_COUNTER
	#include <windows.h>
#elif (PROFILE_CLOCK == PROFILE_GETTIMEOFDAY)
	#include <sys/time.h>
#elif PROFILE_CLOCK == PROFILE_WIN_SYSTEMTIME
	#include <windows.h>
#elif PROFILE_CLOCK == PROFILE_C_CLOCK
	#define PROFILE_CLOCK_RATE CLOCKS_PER_SEC * 1000.0
#endif


#include <vector>
#include <string>



using namespace std;

#define PROFILE_MAX_LEVELS 50
#define PROFILE_LEVEL_INDENT 2

#define pTime double //__int64
//float

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif


class Profile
{
public:
	typedef struct {
		unsigned int queries[2];
	} queryPair;

	/// Contains information about a profiler section
	typedef struct s {
		/** Index of the parent section 
		  * in the previous level
		*/
		int parent;
		/// name of the section
		std::string name;
		/// stores the time when the section starts
		pTime startTime;
		/// query indexes for the beginning 
		/// and end of the section
		std::vector<queryPair> queriesGL[2];

		unsigned long long int totalQueryTime;
		/** wasted time running the 
		  * profiler code for the section
		*/		
		pTime wastedTime;
		/// Total number of calls
		pTime calls;
		/// total time spend in the profiler section
		pTime totalTime;
		/// 
		bool profileGL;
	}section;

	/// Information about a level of profiling
	typedef struct l {
		/// set of sections in this level
		std::vector <section> sec;
		/** stores the current profile section for
		  * each level */
		int cursor;
	}level;


private:


	/// space displacement for dump string formating
	static int sDisp;

	static unsigned int sBackBuffer, sFrontBuffer;

	/// list of all levels
	static level sLevels[PROFILE_MAX_LEVELS];
	/// current level
	static int sCurrLevel;
	/// total number of levels
	static int sTotalLevels;

	/// Puts the profile result in sDump
	static void DumpLevels(int l, int p, pTime calls, std::string &s);



	/// Creates a new section
	void createNewSection(std::string &name, pTime w, bool profileGL);
	/// returns the index of a section
	int searchSection(std::string &name);
	/// updates the times in a section
	void updateSection(int cur, pTime w);
	/// add the time spent in the current section
	void accumulate();

	static void GetTicks(pTime *ticks);

#if PROFILE_CLOCK == PROFILE_WIN_HIGH_PERFORMANCE_COUNTER
	static LARGE_INTEGER sFreq;
#endif

public:
	/// create the profile report and store in s
	static nau_API void DumpLevels(std::string &s);
	/// resets profile data
	static nau_API void Reset();
	/// get profile data
	static nau_API level* GetProfilerData();

	///
	static nau_API void CollectQueryResults();

	/// begin profile section
	nau_API Profile (std::string name, bool profileGL = false);
	/// end profile section
	nau_API ~Profile();


};

#if NAU_PROFILE == NAU_PROFILE_NONE
#define PROFILE(name)
#define PROFILE_GL(name)
#elif NAU_PROFILE == NAU_PROFILE_CPU
#define PROFILE(name) Profile __profile(name)
#define PROFILE_GL(name) Profile __profile(name, false)
#elif NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
#define PROFILE(name) Profile __profile(name)
#define PROFILE_GL(name) Profile __profile(name, true)
#endif
#endif

