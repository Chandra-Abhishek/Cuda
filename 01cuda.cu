#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Bounds of the Mandelbrot set */
#define X_MIN -1.78
#define X_MAX 0.78
#define Y_MIN -0.961
#define Y_MAX 0.961

typedef struct {

  int nb_rows, nb_columns; /* Dimensions */
  char * pixels; /* Linearized matrix of pixels */

} Image;

static void error_options () {

  fprintf (stderr, "Use : ./mandel [options]\n\n");
  fprintf (stderr, "Options \t Meaning \t\t Default val.\n\n");
  fprintf (stderr, "-n \t\t Nb iter. \t\t 100\n");
  fprintf (stderr, "-b \t\t Bounds \t\t -1.78 0.78 -0.961 0.961\n");
  fprintf (stderr, "-d \t\t Dimensions \t\t 1024 768\n");
  fprintf (stderr, "-f \t\t File \t\t /tmp/mandel.ppm\n");
  exit (1);
}

static void analyzis (int argc, char * * argv, int * nb_iter, double * x_min, double * x_max, double * y_min, double * y_max, int * width, int * height, char * * path) {

  const char * opt = "b:d:n:f:" ;
  int c ;

  /* Default values */
  * nb_iter = 500;
  * x_min = X_MIN;
  * x_max = X_MAX;
  * y_min = Y_MIN;
  * y_max = Y_MAX;
  * width = 1024;
  * height = 768;
  * path = "tmp/mandel.ppm";

  /* Analysis of arguments */
  while ((c = getopt (argc, argv, opt)) != EOF) {
    
    switch (c) {
      
    case 'b':
      sscanf (optarg, "%lf", x_min);
      sscanf (argv [optind ++], "%lf", x_max);
      sscanf (argv [optind ++], "%lf", y_min);
      sscanf (argv [optind ++], "%lf", y_max);
      break ;
    case 'd': /* width */
      sscanf (optarg, "%d", width);
      sscanf (argv [optind ++], "%d", height);
      break;
    case 'n': /* Number of iterations */
      * nb_iter = atoi (optarg);
      break;
    case 'f': /* Output file */
      * path = optarg;
      break;
    default :
      error_options ();
    };
  }  
}

static void initialization (Image * im, int nb_columns, int nb_rows) {
  im -> nb_rows = nb_rows;
  im -> nb_columns = nb_columns;
  im -> pixels = (char *) malloc (sizeof (char) * nb_rows * nb_columns); /* Space memory allocation */
} 

static void save (const Image * im, const char * path) {
  /* Image saving using the ASCII format'.PPM' */
  unsigned i;
  FILE * f = fopen (path, "w");  
  fprintf (f, "P6\n%d %d\n255\n", im -> nb_columns, im -> nb_rows); 
  for (i = 0; i < im -> nb_columns * im -> nb_rows; i ++) {
    char c = im -> pixels [i];
    fprintf (f, "%c%c%c", c, c, c); /* Monochrome weight */
  }
  fclose (f);
}

__global__ void kercud(double dx, double dy, char * pixels, int nb_iter, double x_min, double y_max, int num_col){
	   int index_of_X = blockIdx.x * blockDim.x + threadIdx.x;
	   int index_of_Y = blockIdx.y * blockDim.y + threadIdx.y;

	   double a = x_min + index_of_Y *dx, b = y_max - index_of_X * dy, x = 0, y = 0;
	   int i = 0;
	   while (i < nb_iter){
	   	 double tmp = x;
		 x = x * x - y * y + a;
		 y = 2 * tmp * y + b;
		 if (x * x + y * y > 4){
		    break;
		 }
		 else {
		      i ++;
		 }
	   }
	   pixels [index_of_X * num_col + index_of_Y]= (double) i / nb_iter * 255; //formula instead of pos
}

static void compute (Image * im, int nb_iter, double x_min, double x_max, double y_min, double y_max) {
    
  double dx = (x_max - x_min) / im -> nb_columns, dy = (y_max - y_min) / im -> nb_rows; /* Discretization */
  int row_num = im -> nb_rows, num_col = im -> nb_columns;

  dim3 size_of_block(16,16,1);
  dim3 no_of_thrds_in_block(row_num/16, num_col/16, 1);

  char * cuda_pixel;
  cudaMalloc(&cuda_pixel, sizeof(char) * row_num * num_col);
  cudaMemcpy(cuda_pixel, im -> pixels, sizeof(char) * row_num * num_col, cudaMemcpyHostToDevice);

  kercud <<< no_of_thrds_in_block, size_of_block >>> (dx, dy, cuda_pixel, nb_iter, x_min, y_max, num_col);
  cudaDeviceSynchronize();

  cudaMemcpy(im -> pixels, cuda_pixel, sizeof(char) * row_num * num_col, cudaMemcpyDeviceToHost);
  cudaFree(cuda_pixel);
}

int main (int argc, char * * argv) {
  
  int nb_iter, width, height; /* Degree of precision, dimensions of the image */  
  double x_min, x_max, y_min, y_max; /* Bounds of representation */
  char * path; /* File destination */
  Image im;
  analyzis(argc, argv, & nb_iter, & x_min, & x_max, & y_min, & y_max, & width, & height, & path);
  initialization (& im, width, height);
  compute (& im, nb_iter, x_min, x_max, y_min, y_max);
  save (& im, path);

  return 0 ;
}
