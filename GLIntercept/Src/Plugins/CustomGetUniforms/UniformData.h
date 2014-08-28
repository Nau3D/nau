#include <string>
#include "../../MainLib/InterceptPluginInterface.h"
#define CINT 0
#define CUINT 1
#define CFLOAT 2
#define CBOOL 3
#define OTHER -1

using namespace std;

class UniformData
{
public:
	UniformData(int p, int l);
	UniformData(int p, int l, string n, void* d, unsigned int t);

	
	~UniformData();

	void Update(void* d);
	void Update(const char *charptr, FunctionArgs &accessArgs);

	void SetType(unsigned int t);
	void SetTypeManual(unsigned int t, unsigned int c, unsigned int r);

	void SetName(string new_name);
	void SetName(char *new_name);

	bool IsValueSet();

	char *InfoString();
	char *ValueString();
	char *UniformString();
	
protected:
	int program;
	int location;
	string name;
	void *data;
	bool isValueSet;

	unsigned int datatype;
	unsigned int datatype_enum;
	unsigned int datasize[2]; //RxC
private:
	char *outputString;
	unsigned int UniformData::getLength(int number);
	void printRow(ostringstream *os, unsigned int row);
};