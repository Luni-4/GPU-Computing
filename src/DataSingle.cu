#include "DataSingle.h"
#include <fstream>


DataSingle::DataSingle()
{
}


DataSingle::~DataSingle()
{
}

uint8_t* DataSingle::read_file(const char* szFile) {
	std::ifstream file(szFile, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	if (size == -1)
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read((char*)buffer, size);
	return buffer;
}
