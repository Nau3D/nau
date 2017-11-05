#include "profile.h"

#include "nau/slogger.h"

#include <ctime>


#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
#include <glbinding/gl/gl.h>
using namespace gl;
#endif

// Declare static variables
int Profile::sDisp = 0;
int Profile::sCurrLevel = -1;
int Profile::sTotalLevels = 0;
Profile::level Profile::sLevels[PROFILE_MAX_LEVELS];
unsigned int Profile::sBackBuffer = 0, Profile::sFrontBuffer = 1;

#if PROFILE_CLOCK == PROFILE_WIN_HIGH_PERFORMANCE_COUNTER
LARGE_INTEGER Profile::sFreq;
#endif


//int __ProfileFont=(int)GLUT_BITMAP_8_BY_13;

/*
inline void ProfileGetTicks(__int64 *ticks)
{
	__asm
	{
		push edx;
		push ecx;
		mov ecx,ticks;
		_emit 0Fh
		_emit 31h
		mov [ecx],eax;
		mov [ecx+4],edx;
		pop ecx;
		pop edx;
	}
}

*/
void 
Profile::GetTicks(pTime *ticks) {

#if PROFILE_CLOCK == PROFILE_WIN_HIGH_PERFORMANCE_COUNTER
	LARGE_INTEGER t;
	QueryPerformanceCounter(&t) ;
	*ticks = 1000.0*(pTime)((double)t.QuadPart/(double)sFreq.QuadPart);

#elif PROFILE_CLOCK == PROFILE_C_CLOCK
	*ticks =  (pTime)clock()/PROFILE_CLOCK_RATE;

#elif PROFILE_CLOCK == PROFILE_WIN_SYSTEMTIME
	SYSTEMTIME systemTime;
	GetSystemTime( &systemTime );

	FILETIME fileTime;
	SystemTimeToFileTime( &systemTime, &fileTime );

	ULARGE_INTEGER uli;
	uli.LowPart = fileTime.dwLowDateTime; 
	uli.HighPart = fileTime.dwHighDateTime;

	ULONGLONG systemTimeIn_ms( uli.QuadPart/10000 );
	*ticks = (double)systemTimeIn_ms;

#elif PROFILE_CLOCK == PROFILE_GETTIMEOFDAY
	timeval t2;
	gettimeofday(&t2,NULL);
	*ticks = t2.tv_sec * 1000.0 + t2.tv_usec / 1000.0;     
#endif
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////


Profile::Profile(std::string name, bool profileGL) {

	int found;
	pTime w;
	sCurrLevel++;

#if PROFILE_CLOCK == PROFILE_WIN_HIGH_PERFORMANCE_COUNTER
	QueryPerformanceFrequency(&sFreq);
#endif

	GetTicks(&w);

	// create new level
	if (sCurrLevel == sTotalLevels) {

		sLevels[sCurrLevel].cursor = -1;
		createNewSection(name, w, profileGL);
		// store the size of the largest section name
		int aux = (int)name.size() ;
		if (aux > sDisp)
			sDisp = aux;
		sTotalLevels++;
	}
	else {  	
		// search for name and parent
		found = searchSection(name);
		if (found != -1)
			updateSection(found, w);
		else {
			// create new section inside current level
			createNewSection(name, w, profileGL);
			// store the size of the largest section name
			// for report formatting purposes
			int aux = (int)name.size() ;
			if (aux > sDisp)
				sDisp = aux;
		}
	}
}


Profile::~Profile() {
	// add the time spent in the current section
	accumulate();
	// decrease current level
	sCurrLevel--;
}


//////////////////////////////////////////////////////////////////////
// Instance Methods
//////////////////////////////////////////////////////////////////////

void Profile::createNewSection(std::string &name, pTime w, bool profileGL) {

	section s;

#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
	s.profileGL = profileGL;
#else
	s.profileGL = false;
#endif
	s.parent = (sCurrLevel > 0 ? 
		sLevels[sCurrLevel-1].cursor : -1);
	s.name = name;
	s.calls = 1;
	s.totalTime = 0;
	s.totalQueryTime = 0;
	
	sLevels[sCurrLevel].cursor = (int)sLevels[sCurrLevel].sec.size();

#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU	
	if (profileGL) {
		queryPair p;
		glGenQueries(2, p.queries);
		glQueryCounter(p.queries[0], GL_TIMESTAMP);
		s.queriesGL[sBackBuffer].push_back(p);
	}
#endif

	GetTicks(&(s.startTime));
	s.wastedTime = s.startTime - w;
	sLevels[sCurrLevel].sec.push_back(s);
	//SLOG("opening %s", s.name.c_str());
}


int Profile::searchSection(std::string &name) {

	size_t i, max;
	int par;

	max = sLevels[sCurrLevel].sec.size();
	par = (sCurrLevel==0 ? -1 : sLevels[sCurrLevel-1].cursor);

	for(i=0;i<max;i++) {
		if (( name == sLevels[sCurrLevel].sec[i].name)  && (par == sLevels[sCurrLevel].sec[i].parent))
			return((int)i);
	}
	return(-1);
}

void Profile::updateSection(int cur, pTime w) {

	section *s;

	s = &(sLevels[sCurrLevel].sec[cur]);
	s->calls++;
	sLevels[sCurrLevel].cursor = cur;

#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
	if (s->profileGL) {
		queryPair p;
		glGenQueries(2, p.queries);
		glQueryCounter(p.queries[0], GL_TIMESTAMP);
		s->queriesGL[sBackBuffer].push_back(p);
	}
#endif

	GetTicks(&s->startTime);
	s->wastedTime += s->startTime - w;
}

void Profile::accumulate() {

	section *s;
	pTime t,t2;
	GetTicks(&t);

	s = &(sLevels[sCurrLevel].sec[sLevels[sCurrLevel].cursor]);

#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
	if (s->profileGL) { 
		glQueryCounter(s->queriesGL[sBackBuffer][s->queriesGL[sBackBuffer].size()-1].queries[1], GL_TIMESTAMP);
	}
#endif

	// to measure wasted time when accumulating
	GetTicks(&t2);
	s->wastedTime += (t2-t);
	s->totalTime += (t - s->startTime);
	//SLOG("closing %s", s->name.c_str());
}

//////////////////////////////////////////////////////////////////////
// Class Methods
//////////////////////////////////////////////////////////////////////


void Profile::Reset() {

	for(int i=0; i < sTotalLevels; ++i) {

#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
		for (unsigned int s = 0; s < sLevels[i].sec.size(); ++s) {
			for (unsigned int k = 0; k < sLevels[i].sec[s].queriesGL[0].size(); ++k) {
				glDeleteQueries(2, sLevels[i].sec[s].queriesGL[0][k].queries);
			}
			for (unsigned int k = 0; k < sLevels[i].sec[s].queriesGL[1].size(); ++k) {
				glDeleteQueries(2, sLevels[i].sec[s].queriesGL[1][k].queries);
			}
		}
#endif

		sLevels[i].sec.clear();
		sLevels[i].cursor = 0;
	}
	
	sTotalLevels = 0; 
	sBackBuffer = 0;
	sFrontBuffer = 1;
	sCurrLevel = -1;
	sDisp = 0;

}


void Profile::DumpLevels(int l, int p, pTime calls, std::string &dump) {

	size_t siz;
	char a[2] = "";
	char s[200];
	char s2[200];
	section *sec;

	siz = sLevels[l].sec.size();

	for(size_t cur = 0; cur < siz; ++cur) {
		sec = &(sLevels[l].sec[cur]);

		if (l==0)
			calls = sec->calls;

		if ((p == -1) || (sec->parent == p)) {

			sprintf(s,"%*s%s", l * PROFILE_LEVEL_INDENT ," ",sec->name.c_str());

			if (sec->profileGL)
				sprintf(s2,"%-*s %5.0f %8.2f %8.2f %8.2f\n",
					sDisp + sTotalLevels * PROFILE_LEVEL_INDENT + 2,
					s,
					(float)(sec->calls/calls),
					(float)(sec->wastedTime)/(calls), (sec->totalQueryTime/(1000000.0 * calls)),
					(float)(sec->totalTime)/(calls));
			else
				sprintf(s2,"%-*s %5.0f %8.2f          %8.2f\n",
					sDisp + sTotalLevels * PROFILE_LEVEL_INDENT + 2,
					s,
					(float)(sec->calls/calls),
					(float)(sec->totalTime)/(calls),
					(float)(sec->wastedTime)/(calls));

			dump += s2;

			if (l+1 < sTotalLevels)
				DumpLevels(l+1,(int)cur,calls, dump);
		}
		
	}
}

void
Profile::DumpLevels(std::string &dump ) {

#if NAU_PROFILE != NAU_PROFILE_NONE
	int indent = sTotalLevels * PROFILE_LEVEL_INDENT + sDisp;
	char saux[100];
		
	char t1[5]="Name";
	char t2[7]="#c";
	char t3[9]="#tc";
	char t4[8]="CPU(ms)";
	char t41[8] ="GPU(ms)";
	char t5[3]="wt";

	dump = "";
	sprintf(saux,"%-*s  %s  %s  %s       %s\n",indent+4,t1,t2,t4,t41,t5);
	dump += saux; 
	sprintf(saux,"---- %*s\n",indent+31,"------------------------------------");
	dump += saux;

	DumpLevels(0,-1,sLevels[0].sec[0].calls, dump);
#else
	dump = "";
#endif

}


void 
Profile::CollectQueryResults() {

#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
	size_t siz;
	section *sec;
	int availableEnd = 0;
	GLuint64 timeStart=0, timeEnd = 0;
	unsigned long long int aux = 0;

	for (int l = 0; l < sTotalLevels; ++l) {
		siz = sLevels[l].sec.size();

		for(unsigned int cur = 0; cur < siz; ++cur) {
			sec = &(sLevels[l].sec[cur]);

			if (sec->profileGL) {

				aux = 0;

				for (unsigned int j = 0; j < sec->queriesGL[sFrontBuffer].size(); ++j) {

					glGetQueryObjectui64v(sec->queriesGL[sFrontBuffer][j].queries[0], GL_QUERY_RESULT, &timeStart);
					glGetQueryObjectui64v(sec->queriesGL[sFrontBuffer][j].queries[1], GL_QUERY_RESULT, &timeEnd);
					aux +=  (timeEnd - timeStart);
					glDeleteQueries(2, sec->queriesGL[sFrontBuffer][j].queries);
				}
				sec->totalQueryTime += aux;
				sec->queriesGL[sFrontBuffer].clear();
			}
		}
	}
	// SWAP QUERY BUFFERS
	if (sBackBuffer) {
		sBackBuffer = 0;
		sFrontBuffer = 1;
	}
	else {
		sBackBuffer = 1;
		sFrontBuffer = 0;
	}
#endif
}


