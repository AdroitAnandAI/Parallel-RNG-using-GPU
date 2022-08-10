// Some code obtained from the presentation file below.0
// http://www.acclab.helsinki.fi/~aakurone/tl3/lecturenotes/09_random_numbers-1x2.pdf

#include <stdio.h>
#include <string>
// #include <time.h>
#include "sys/time.h"

using namespace std;

// R250 RNG where r = 250 and q = 103

#define NR 1000
#define NR250 1250
#define NRp1 1001
#define NR250p1 1251

int getseed()
{
    int i;
    struct timeval tp;
    if (gettimeofday(&tp,(struct timezone *)NULL) == 0) {
        i=tp.tv_sec+tp.tv_usec;
        i=(i%1000000)|1;
        return i;
    } else {
        return -1;
    }
}

double lcgy(int *seed) {
    static int a=16807, m=2147483647,
    q=127773, r=2836;
    double minv = (double) 1.0/m;
    *seed = a*(*seed % q)-r*(*seed / q);
    if (*seed < 0) *seed = *seed + m;
    return (double) *seed * minv;
}

void r250(int *x,double *r,int n)
{
    static int q=103,p=250;
    static double rmaxin=2147483648.0; /* 2**31 */
    int i,k;
    for (k=1;k<=n;k++) {
        x[k+p]=x[k+p-q]^x[k];
        r[k]=(double)x[k+p]/rmaxin;
    }
    for (i=1;i<=p;i++) x[i]=x[n+i];
}
double ran_number(int *seed)
{
    double ret;
    static int firsttime=1;
    static int i,j=NR;
    static int x[NR250p1];
    static double r[NRp1];
    if (j>=NR) {
        if (firsttime==1) {
            for (i=1;i<=250;i++)
                x[i]=2147483647.0*lcgy(seed);
            firsttime=0;
        }
        r250(x,r,NR);
        j=0;
    }
    j++;
    return r[j];
}



// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, unsigned long long int *matrix, int rows, int cols, int skip) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            if ((i*cols+j) % (skip + 1) == 0) {
                fprintf(outputFilePtr, "%llu ", matrix[i*cols+j]);
                fprintf(outputFilePtr, "\n");
            }
        }       
    }
}



int main(int argc, char **argv)
{
    printf("test1");
    int seed;
    int i,imax;
    double x,y;


    FILE *outputFilePtr;

    imax=atoi(*++argv);
    seed=atoi(*++argv);
    printf("test1");
    if (seed<=0) seed=getseed();
    fprintf(stderr,"Seed %d\n",seed);
    printf("test2");
    outputFilePtr = fopen("gfsrout.txt", "w");

    for(i=0; i<imax; i++) {
        x=ran_number(&seed); //y=ran_number(&seed);
        // fprintf(stdout,"%g %g\n",x,y);
        fprintf(outputFilePtr, "%d ", int(x*2147483648));
        fprintf(outputFilePtr, "\n");        
    }

    fclose(outputFilePtr);
    return(0);
}