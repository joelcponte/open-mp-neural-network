#include "functions.h"

int a1_size = 20;
int output_size = 2;
int frows = 1200;
int fcols = 60;
int test_set_rows = 41;
int NUM_EPOCHS = 1000;
double lr = 0.001; // learning rate


int main()
{
	srand(10);

	int TEST = 1;
	int PRINT_WEIGHTS = 0;

	double** x_train;
	double** x_test;
	double** y_train;
	double** y_test;
	double ** w1;
	double** w2;
	
	double* b1 = new double[a1_size]();
	double* b2 = new double[output_size]();

	//////// WEIGHTS INITIALIZATION

	w1 = alloc_matrix(w1, fcols, a1_size);
	w2 = alloc_matrix(w2, a1_size, output_size);


	int number_of_threads = omp_get_max_threads();
	int size_each_batch = floor(frows/number_of_threads);


	x_train = alloc_matrix(x_train, frows, fcols);
	y_train = alloc_matrix(y_train, frows, 2);
	x_test = alloc_matrix(x_test, test_set_rows, fcols);
	y_test = alloc_matrix(y_test, test_set_rows, 2);
	
	//////// IMPORT TRAINING AND TEST SETS

	char y_train_filename[] = "y_train.csv";
	char x_train_filename[] = "x_train.csv";
	char x_test_filename[] = "x_test.csv";
	char y_test_filename[] = "y_test.csv";

	x_train = read_file(x_train, frows, fcols, x_train_filename);
	x_test = read_file(x_test, test_set_rows, fcols, x_test_filename);
	y_train = read_file(y_train, frows, 2, y_train_filename);
	y_test = read_file(y_test, test_set_rows, 2, y_test_filename);

	cout << "\nFINISHED READING FILES...." << "\n\n";

	double loss;

	//////// WEIGHTS INITIALIZATION

	w1 = initialize_Weights(w1, fcols, a1_size);
	w2 = initialize_Weights(w2, a1_size, output_size);

	#pragma omp parallel default(shared) num_threads(number_of_threads)
	{

		double ** a1;
		double ** a2;
		double ** z1;
		double ** z2;

		double ** dw1;
		double ** dw2;
		double ** dz1;
		double ** dz2;

		//////// ALLOC ALL VARIABLES

		a1 = alloc_matrix(a1, size_each_batch, a1_size);
		z1 = alloc_matrix(z1, size_each_batch, a1_size);
		z2 = alloc_matrix(z2, size_each_batch, output_size);
		a2 = alloc_matrix(a2, size_each_batch, output_size);
		dz2 = alloc_matrix(dz2, size_each_batch, output_size);
		dw2 = alloc_matrix(dw2, a1_size, output_size);
		dz1 = alloc_matrix(dz1, size_each_batch, a1_size);
		dw1 = alloc_matrix(dw1, fcols, a1_size);
		double* db2 = new double[output_size]();
		double* db1 = new double[a1_size]();


		//////// TRAIN

		int l = omp_get_thread_num();

		for(int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
			// #pragma omp master
			// loss = 0;

			
			//////// FORWARD PROPAGATION	

			for(int i = 0; i < size_each_batch; ++i) {
				for(int j = 0; j < a1_size; ++j) {
					z1[i][j] = 0;
					for(int k = 0; k < fcols; ++k) {
						z1[i][j] += x_train[i + l*size_each_batch][k] * w1[k][j] + b1[j];
					}
					a1[i][j] = tanh(z1[i][j]);
					// if ( i == 0 & l == 0) cout << x_train_current_batch[i][0] << endl;
				}
			}


			for(int i = 0; i < size_each_batch; ++i) {
				for(int j = 0; j < output_size; ++j) {
					z2[i][j] = 0;
					for(int k = 0; k < a1_size; ++k) {
						z2[i][j] += a1[i][k] * w2[k][j] + b2[j];
					}
					a2[i][0] = exp(z2[i][0])/(exp(z2[i][0]) + exp(z2[i][1]));
					a2[i][1] = exp(z2[i][1])/(exp(z2[i][0]) + exp(z2[i][1]));
				}
			}
			//

			//////// BACKPROPAGATION

			for(int i = 0; i < size_each_batch; ++i){
				for(int j = 0; j < output_size; ++j){
					dz2[i][j] = a2[i][j]-y_train[i + l*size_each_batch][j];
				}
			}


			for(int i = 0; i < a1_size; ++i){
				for(int j = 0; j < output_size; ++j) {
					dw2[i][j] = 0;
					db2[j] = 0;
					for(int k = 0; k < size_each_batch; ++k) {
						dw2[i][j] += dz2[k][j]*a1[k][i];
						db2[j] += dz2[k][j];
					}
				}
			}

			for(int i = 0; i < size_each_batch; ++i){
				for(int j = 0; j < a1_size; ++j) {
					dz1[i][j] = 0;
					for(int k = 0; k < output_size; ++k)
						dz1[i][j] += dz2[i][k]*w2[j][k]*(1 - a1[i][j]*a1[i][j]);
				}
			}

			for(int i = 0; i < fcols; ++i){
				for(int j = 0; j < a1_size; ++j) {
					dw1[i][j] = 0;
					db1[j] = 0;
					for(int k = 0; k < size_each_batch; ++k) {
						dw1[i][j] += dz1[k][j]*x_train[k + l*size_each_batch][i];
						db1[j] += dz1[k][j];
					}
				}
			}
			//

			//////// CALCULATE THE LOCAL CONTRIBUTION OF THE LOSS TO THE TOTAL LOSS

			double loss_loc =0. ;
			for(int i = 0; i < size_each_batch; ++i) {
				loss_loc -= y_train[i + l*size_each_batch][0]*log(a2[i][0]) +  y_train[i + l*size_each_batch][1]*log(a2[i][1]);
			}


			//////// GRADIENT DESCEND

			#pragma omp barrier

			#pragma omp critical
			{
				// cout << epoch << "\t" << loss_loc << endl;
				loss += loss_loc;

				for(int i = 0; i < a1_size; ++i) {
					for(int j = 0; j < output_size; ++j) {
						w2[i][j] -= lr*dw2[i][j];
					}
				}
				for(int i = 0; i < fcols; ++i) {
					for(int j = 0; j < a1_size; ++j) {
						w1[i][j] -= lr*dw1[i][j];
					}
				}

				for(int j = 0; j < a1_size; ++j) {
					b1[j] -= lr*db1[j];
				}

				for(int j = 0; j < output_size; ++j) { 
					b2[j] -= lr*db2[j];
				}
			}
			// #pragma omp barrier
			// #pragma omp master
			// cout << epoch << "\t" << loss/(frows) << endl;
		} // END EPOCH LOOP
	}


	//////// TEST ON NEW DATA
	// UNEXPENSIVE... NOT PARALLELIZED
	double ** a1test;
	double ** a2test;
	double ** z1test;
	double ** z2test;
	a1test = alloc_matrix(a1test, test_set_rows, a1_size);
	z1test = alloc_matrix(z1test, test_set_rows, a1_size);
	z2test = alloc_matrix(z2test, test_set_rows, output_size);
	a2test = alloc_matrix(a2test, test_set_rows, output_size);



	//////// FORWARD PROP ON TEST SET

	for(int i = 0; i < test_set_rows; ++i) {
		for(int k = 0; k < a1_size; ++k) {
			z1test[i][k] = 0;
			for(int j = 0; j < fcols; ++j) {
				z1test[i][k] += x_test[i][j] * w1[j][k] + b1[k];
			}
			a1test[i][k] = tanh(z1test[i][k]);
		}
	}


	for(int i = 0; i < test_set_rows; ++i) {
		for(int k = 0; k < output_size; ++k) {
			z2test[i][k] = 0;
			for(int j = 0; j < a1_size; ++j) {
				z2test[i][k] += a1test[i][j] * w2[j][k] + b2[k];
			}
			a2test[i][0] = exp(z2test[i][0])/(exp(z2test[i][0]) + exp(z2test[i][1]));
			a2test[i][1] = exp(z2test[i][1])/(exp(z2test[i][0]) + exp(z2test[i][1]));
		}
	}


	y_test = read_file(y_test, test_set_rows, 2, y_test_filename);

	//////// CHECK ACCURACY

	double correct_test = 0;

	for (int i = 0; i < test_set_rows; ++i) {
		if (y_test[i][0] == 1) {
			if (a2test[i][0] >= 0.5)
				correct_test++;
		} else {
			if (a2test[i][1] > 0.5)
				correct_test++;
		}
	}

	cout << "Test set accuracy: " <<  correct_test/test_set_rows << "\n";

	return 0;
}