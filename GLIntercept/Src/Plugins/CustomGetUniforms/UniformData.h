#include <string>

using namespace std;

class UniformData
{
public:
	UniformData(int p, int l);
	UniformData(int p, int l, string n, void* d, unsigned int t);

	
	~UniformData();

	void Update(void* d);

	void SetType(unsigned int t);

	void SetName(string new_name);
	void SetName(char *new_name);

	char *InfoString();
	char *ValueString();
	
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