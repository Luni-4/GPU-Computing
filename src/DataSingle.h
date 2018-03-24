#ifdef _WIN32
#include "Windows.h"
#endif

#include <cstdint>

class DataSingle {

public:
	DataSingle();
	~DataSingle();

	uint8_t* read_file(const char* szFile);

};

