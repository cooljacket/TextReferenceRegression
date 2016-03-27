#include <stdio.h>
#include <ctime>
#include "LinearRegression.h"

int main() {
	clock_t start = clock();
	LinearRegression lg("train.csv", 0.095, 10000);
	lg.SGDTrain(2000);
	printf("Totol cost time %lfs\n", (clock()-start)*1.0/CLOCKS_PER_SEC);
	// vector<double> result(lg.Test());
	// do something more about the predict result...

	return 0;
}