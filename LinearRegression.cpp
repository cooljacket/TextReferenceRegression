#include "LinearRegression.h"
#include <string>
#include <sstream>
#include <stdio.h>
#include <ctime>
#include <algorithm>
using namespace std;

LinearRegression::LinearRegression(char* fileName, double alpha, int MAX_Iter, double costBound)
: alpha(alpha), costBound(costBound), Max_Iteration(MAX_Iter)
{
	fstream data(fileName, ios::in);
	readData(data, train_x, true);
	data.close();
	delta = vector<double>(train_x.size());
	theta = vector<double>(train_x[0].size());
}


void LinearRegression::readData(fstream& in, DATA& X, bool isTrain) {
	string line;
	int id;
	double value;
	vector<double> x(ColSize+1, 1.0);
	char buf;
	clock_t start = clock();

	getline(in, line);
	printf("Reading...\n");
	while (getline(in, line)) {
		if (line.empty())
			continue;
		stringstream ss(line);
		ss >> id;
		for (int i = 1; i <= ColSize; ++i)
			ss >> buf >> x[i];
		if (isTrain) {
			ss >> buf >> value;
			train_y.push_back(value);
		}
		X.push_back(x);
	}
	printf("Total %d samples, use time %lfs\n", int(X.size()), (clock()-start)*1.0/CLOCKS_PER_SEC);
}


double LinearRegression::evaluate(const vector<double>& X, int index) {
	double h = 0;
	for (int i = 0; i < theta.size(); ++i)
		h += theta[i] * X[i];
	return h;
}


void LinearRegression::calDelta(const DATA& X, const vector<double>& Y) {
	delta = vector<double>(X.size());
	for (int i = 0; i < X.size(); ++i)
		delta[i] += evaluate(X[i], i) - Y[i];
}


void LinearRegression::gradientDescent(const DATA& X) {
	for (int i = 0; i < theta.size(); ++i) {
		double sum = 0.0;
		for (int j = 0; j < X.size(); ++j)
			sum += delta[j] * X[j][i];
		theta[i] -= alpha / X.size() * sum;
	}
}


double LinearRegression::cost() {
	double J = 0.0;
	for (int i = 0; i < delta.size(); ++i)
		J += delta[i] * delta[i];
	return J / (delta.size()*2.0);
}


double LinearRegression::trainHelper(const DATA& X, const vector<double>& Y) {
	calDelta(X, Y);
	gradientDescent(X);
}


double LinearRegression::BatchTrain() {
	int times = 0;
	clock_t start = clock();
	calDelta(train_x, train_y);

	while (times++ < Max_Iteration) {
		double J = cost();
		printf("Times[%d], cost=%.6lf, use time %lfs\n", times, J, (clock()-start)*1.0/CLOCKS_PER_SEC);
		start = clock();
		if (J < costBound)
			break;
		trainHelper(train_x, train_y);
	}

	calDelta(train_x, train_y);
	double J = cost();
	printf("Finally, cost=%.6lf\n", J);
	return J;
}


void LinearRegression::sgdHelper(int mini_batch_size) {
	int total_size = train_x.size();
	vector<int> select(total_size);
	for (int i = 0; i < select.size(); ++i)
		select[i] = i;
	random_shuffle(select.begin(), select.end());

	for (int i = 0; i < total_size; i += mini_batch_size) {
		int size = min(mini_batch_size, total_size-i);
		DATA X(size);
		vector<double> Y(size);
		for (int j = 0; j < size; ++j) {
			X[j] = train_x[select[i+j]];
			Y[j] = train_y[select[i+j]];
		}
		trainHelper(X, Y);
	}
}


double LinearRegression::SGDTrain(int mini_batch_size) {
	int times = 0;
	clock_t start = clock();
	calDelta(train_x, train_y);
	double Min_Cost = cost();
	vector<double> best_theta;

	while (times++ < Max_Iteration) {
		double J = cost();
		if (J < Min_Cost) {
			Min_Cost = J;
			best_theta = theta;
		}
		printf("Times[%d], cost=%.3lf/best=%.3lf, use time %lfs\n", times, J, Min_Cost, (clock()-start)*1.0/CLOCKS_PER_SEC);
		start = clock();
		if (J < costBound)
			break;
		sgdHelper(mini_batch_size);
	}

	calDelta(train_x, train_y);
	double J = cost();
	printf("Finally, cost=%.3lf/best=%.3lf\n", J, Min_Cost);
	return J;
}


vector<double> LinearRegression::Test() {
	fstream test("test.csv", ios::out);
	readData(test, test_x, false);

	vector<double> predictY(test_x.size());
	for (int i = 0; i < test_x.size(); ++i)
		predictY[i] = evaluate(test_x[i], i);
	return predictY;
}
