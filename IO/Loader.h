#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "Test.h"
#include "TestMulti.h"
#include "Tokanizer.h"
#include "processing.h"
#include "Filter.h"
#include "Factory.h"
#include "processing.h"

using namespace std;
using namespace processing;

inline void insertLineFromStreamIntoTokanizer(ifstream& stream, Tokanizer& tokanizer)
{
	static string line;
	bool result;
	if (getline(stream, line))
	{
		cout << line << endl;
		tokanizer.setText(line);
	}
	else
	{
		throw std::runtime_error("Unexpected end of file");
	}
}

inline void readLine(ifstream& stream) 
{
	static string line;
	if (getline(stream, line))
	{
		cout << line << endl;
	}
	else
	{
		throw std::runtime_error("Unexpected end of file");
	}
}

class Loader
{
public:
	Loader(const Loader& other) = delete;
	Loader& operator=(const Loader& other) = delete;

	static void load(const string& filename);
	




private:
	Loader() {};




};

