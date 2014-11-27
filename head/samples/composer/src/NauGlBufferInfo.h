class NauGlBufferInfo{
private:
	bool isVAO;
public:
	int index;
	int size;
	int components;
	int type;
	int stride;
	int normalized;
	int divisor;
	int integer;

	NauGlBufferInfo();
	NauGlBufferInfo(int indexp, int sizep);
	NauGlBufferInfo(int indexp, int sizep, int componentsp, int typep, int stridep, int normalizedp, int divisorp, int integerp);



	bool isVAOBuffer();
};