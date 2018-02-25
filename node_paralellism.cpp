#include "functions.h"

int a1_size = 20;
int output_size = 2;
int frows = 1200;
int fcols = 60;
int test_set_rows = 41;
int NUM_EPOCHS = 1000;
double lr = 0.001; // learning rate


int main() {
	srand(10);

	double** x_train;
	double** x_test;
	double** y_train;
	double** y_test;

	double ** w1;
	double ** w2;
	double ** a1;
	double ** a2;
	double ** z1;
	double ** z2;

	double ** dw1;
	double ** dw2;
	double ** dz1;
	double ** dz2;

	double ** a2_test;
	double loss;
	int i,j,k;

	//////// ALLOC ALL VARIABLES

	w1 = alloc_matrix(w1, fcols, a1_size);
	a1 = alloc_matrix(a1, frows, a1_size);
	z1 = alloc_matrix(z1, frows, a1_size);
	w2 = alloc_matrix(w2, a1_size, output_size);

	double* b1 = new double[a1_size]();
	double* b2 = new double[output_size]();

	z2 = alloc_matrix(z2, frows, output_size);
	a2 = alloc_matrix(a2, frows, output_size);
	a2_test = alloc_matrix(a2_test, test_set_rows, output_size);
	dz2 = alloc_matrix(dz2, frows, output_size);
	dw2 = alloc_matrix(dw2, a1_size, output_size);
	dz1 = alloc_matrix(dz1, frows, a1_size);
	dw1 = alloc_matrix(dw1, fcols, a1_size);


	double* db2 = new double[output_size]();
	double* db1 = new double[a1_size]();

	x_train = alloc_matrix(x_train, frows, fcols);
	y_train = alloc_matrix(y_train, frows, 2);
	x_test = alloc_matrix(x_test, test_set_rows, fcols);
	y_test = alloc_matrix(y_test, test_set_rows, 2);

	//////// IMPORT DATA

	char y_train_filename[] = "y_train.csv";
	char x_train_filename[] = "x_train.csv";
	char x_test_filename[] = "x_test.csv";
	char y_test_filename[] = "y_test.csv";

	x_train = read_file(x_train, frows, fcols, x_train_filename);
	x_test = read_file(x_test, test_set_rows, fcols, x_test_filename);
	y_train = read_file(y_train, frows, 2, y_train_filename);
	y_test = read_file(y_test, test_set_rows, 2, y_test_filename);

	cout << "\nFINISHED READING FILES...." << "\n\n";
	// omp_set_num_threads(8);

	//////// WEIGHTS INITIALIZATION

	w1 = initialize_Weights(w1, fcols, a1_size);
	w2 = initialize_Weights(w2, a1_size, output_size);

	
	//////// TRAIN

	for(int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {

		//////// FORWARD PROPAGATION	

		#pragma omp parallel for private(j,k)
		for(i = 0; i < frows; ++i) {
			for(j = 0; j < a1_size; ++j) {
				z1[i][j] = 0;
				for(k = 0; k < fcols; ++k) {
					z1[i][j] += x_train[i][k] * w1[k][j] + b1[j];
				}
				a1[i][j] = tanh(z1[i][j]);
			}
		}


		#pragma omp parallel for private(j,k)
		for(i = 0; i < frows; ++i) {
			for(j = 0; j < output_size; ++j) {
				z2[i][j] = 0;
				for(k = 0; k < a1_size; ++k) {
					z2[i][j] += a1[i][k] * w2[k][j] + b2[j];
				}
			}
			a2[i][0] = exp(z2[i][0])/(exp(z2[i][0]) + exp(z2[i][1]));
			a2[i][1] = exp(z2[i][1])/(exp(z2[i][0]) + exp(z2[i][1]));
		}
		//


		//////// BACKPROPAGATION

		#pragma omp parallel for private(j)
		for(i = 0; i < frows; ++i) {
			for(j = 0; j < output_size; ++j) {
				dz2[i][j] = a2[i][j]-y_train[i][j];
			}
		}

		#pragma omp parallel for private(i, k)
		for(j = 0; j < output_size; ++j) {
			for(i = 0; i < a1_size; ++i) {
				dw2[i][j] = 0;
				db2[j] = 0;
				for(k = 0; k < frows; ++k) {
					dw2[i][j] += dz2[k][j]*a1[k][i];
					db2[j] += dz2[k][j];
				}
			}
		}

		#pragma omp parallel for private(j, k)
		for(i = 0; i < frows; ++i) {
			for(j = 0; j < a1_size; ++j) {
				dz1[i][j] = 0;
				for(k = 0; k < output_size; ++k) {
					dz1[i][j] += dz2[i][k]*w2[j][k]*(1 - a1[i][j]*a1[i][j]);
				}
			}
		}

		#pragma omp parallel for private(i, k)
		for(j = 0; j < a1_size; ++j) {
			for(i = 0; i < fcols; ++i) {
				dw1[i][j] = 0;
				db1[j] = 0;
				for(k = 0; k < frows; ++k) {
					dw1[i][j] += dz1[k][j]*x_train[k][i];
					db1[j] += dz1[k][j];
				}
			}
		}
		//

		//////// CALCULATE LOSS

		loss = 0;
		#pragma omp parallel for reduction(+:loss)
		for(int i = 0; i < frows; ++i) {
			loss -= y_train[i][0]*log(a2[i][0]) +  y_train[i][1]*log(a2[i][1]);
		}

		//////// GRADIENT DESCEND

		#pragma omp parallel for private(j)
		for(int i = 0; i < a1_size; ++i) {
			for(int j = 0; j < output_size; ++j) {
				w2[i][j] -= lr*dw2[i][j];
			}
		}

		#pragma omp parallel for private(j)
		for(int i = 0; i < fcols; ++i){
			for(int j = 0; j < a1_size; ++j) {
				w1[i][j] -= lr*dw1[i][j];
			}
		}

		#pragma omp parallel for
		for(int j = 0; j < a1_size; ++j) {
			b1[j] -= lr*db1[j];
		}

		#pragma omp parallel for
		for(int j = 0; j < output_size; ++j) {
			b2[j] -= lr*db2[j];
		}
	} // END EPOCH LOOP


	//////// TEST ON NEW DATA

	// #pragma omp parallel for private(j,k) 
	for(int i = 0; i < test_set_rows; ++i) {
		for(int k = 0; k < a1_size; ++k) {
			z1[i][k] = 0;
			for(int j = 0; j < fcols; ++j) {
				z1[i][k] += x_test[i][j] * w1[j][k] + b1[k];
			}
			a1[i][k] = tanh(z1[i][k]);
		}
	}

	// #pragma omp parallel for private(j,k)
	for(int i = 0; i < test_set_rows; ++i) {
		for(int k = 0; k < output_size; ++k) {
			z2[i][k] = 0;
			for(int j = 0; j < a1_size; ++j) {
				z2[i][k] += a1[i][j] * w2[j][k] + b2[k];
			}
			a2_test[i][0] = exp(z2[i][0])/(exp(z2[i][0]) + exp(z2[i][1]));
			a2_test[i][1] = exp(z2[i][1])/(exp(z2[i][0]) + exp(z2[i][1]));
		}
	}
	y_test = read_file(y_test, test_set_rows, 2, y_test_filename);

	/////// CHECK ACCURACY

	double correct_test = 0;

	// #pragma omp parallel for reduction(+:correct_test)
	for (int i = 0; i < test_set_rows; ++i) {
		if (y_test[i][0] == 1) {
			if (a2_test[i][0] >= 0.5)
				correct_test++;
		} else {
			if (a2_test[i][1] > 0.5)
				correct_test++;
		}
	}

	cout << "Test set accuracy: " <<  correct_test/test_set_rows << "\n";


	return 0;
}
