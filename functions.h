#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <omp.h>
using namespace std;

void print_matrix(double** matrix, int row, int col);
void print_vector(double* array_, int numberOfLines);
double sigmoid(double x);
double ReLU(double x);
double** read_file(double ** data, int frows, int fcols, char* filename);
double ** alloc_matrix(double** matrix, int nrows, int ncols);
double ** initialize_Weights(double** matrix, int nrows, int ncols);


double sigmoid(double x) {
	return(1./(1 + exp(-x)));
}


double** read_file(double ** data, int frows, int fcols, char* filename) {

	ifstream file(filename);
	for(int row = 0; row < frows; ++row) {
		std::string line;
		std::getline(file, line);
		if ( !file.good() ) 
			break;

		std::stringstream iss(line);

		for (int col = 0; col < fcols; ++col) {
			std::string val;
			std::getline(iss, val, ',');


			std::stringstream convertor(val);


			convertor >> data[row][col];

		}
	}
	return(data);
}

double ** alloc_matrix(double** matrix, int nrows, int ncols) {

	matrix = new double*[nrows];

	for (int i = 0; i < nrows; ++i) {
		matrix[i] = new double[ncols]();
	}

	return(matrix);

}

double ** initialize_Weights(double** matrix, int nrows, int ncols) {

	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j){
			matrix[i][j] = 1.*rand()/RAND_MAX;
		}
	}

	return(matrix);
}