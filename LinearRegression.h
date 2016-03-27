#ifndef __LINEAR_REGRESSION_H__
#define __LINEAR_REGRESSION_H__

#include <vector>
#include <fstream>
using namespace std;

typedef vector<vector<double> > DATA;

class LinearRegression
{
	static const int ColSize = 384;
	vector<double> delta, theta;
	DATA train_x, test_x;
	vector<double> train_y;
	double alpha, costBound;
	int Max_Iteration;

public:
	LinearRegression(char* fileName, double alpha, int MAX_Iter=50, double costBound=0.0001);
	void readData(fstream& in, DATA& X, bool isTrain);
	double evaluate(const vector<double>& X, int index);
	void calDelta(const DATA& X, const vector<double>& Y);
	void gradientDescent(const DATA& X);
	double cost();
	double trainHelper(const DATA& X, const vector<double>& Y);
	double BatchTrain();
	void sgdHelper(int mini_batch_size);
	double SGDTrain(int mini_batch_size=1000);
	vector<double> Test();
};

#endif