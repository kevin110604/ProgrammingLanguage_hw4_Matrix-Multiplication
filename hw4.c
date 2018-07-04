/*
 * Title: matrix multiplication with parallel
 * Author: kevin110604
 * Compile: gcc -fopenmp hw4.c -o hw4
 * Run: ./hw4 test_file
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>                                //required by OpenMP


void Strassen(int n, int C[][n], int A[][n], int B[][n]);
void matrix_multiply_it(int n, int C[][n], int A[][n], int B[][n]);
void matrix_add(int n, int C[][n], int A[][n], int B[][n]);
void matrix_sub(int n, int C[][n], int A[][n], int B[][n]);
void matrix_input(int n, int x[][n]);
void matrix_output(int n, int x[][n]);
void Strassen_v2(int n, int C[][n], int A[][n], int B[][n]);


int main(int argc, char *argv[])
{
    int n;                                      //matrix size n*n
    double it_time, st_time;                    //native method time, Strassen's method time
    clock_t start, end;
    FILE *fp;

    if (argc < 2) {
         fprintf(stderr,"ERROR, no input file provided\n");
         exit(1);
    }
    fp=fopen(argv[1], "r");
    fscanf(fp, "%d%d", &n, &n);
    puts("read n.........done");
    printf("n=%d...........done\n", n);
    

    int (*a)[n], (*b)[n], (*c_it)[n], (*c_st)[n];
    a=malloc(n*sizeof(*a));
    b=malloc(n*sizeof(*b));
    c_it=malloc(n*sizeof(*c_it));
    c_st=malloc(n*sizeof(*c_st));
    puts("malloc.........done");
    

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
           fscanf(fp, "%d", &a[i][j]);
        }        
    }
    puts("read a[n][n]...done");
    fscanf(fp, "%d%d", &n, &n);
    puts("read n.........done");
    printf("n=%d...........done\n", n);
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
           fscanf(fp, "%d", &b[i][j]);
        }        
    }
    puts("read b[n][n]...done");
    
    start=clock();
    matrix_multiply_it(n, c_it, a, b);
    end=clock();
    it_time=(double) (end-start)/CLOCKS_PER_SEC;
    printf("The run time using traditional method is %f s\n", it_time);
    
    start=clock();
    Strassen(n, c_st, a, b);
    end=clock();
    st_time=(double) (end-start)/CLOCKS_PER_SEC;
    printf("The run time using Strassen's  method is %f s\n", st_time);


    //matrix_output(n, c_it);
    //matrix_output(n, c_st);
    
    //check if the two answers are the same
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (c_st[i][j]==c_it[i][j])
                ;
            else {
                puts("WRONG!!");
                break;
            }
        }
    }
    

    free(a);
    free(b);
    free(c_it);
    free(c_st);

    return 0;
}

void matrix_multiply_it(int n, int C[][n], int A[][n], int B[][n])
{
    #pragma omp parallel for
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            C[i][j]=0;
            for (int k=0; k<n; k++) {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

void Strassen(int n, int C[][n], int A[][n], int B[][n]) 
{
    if (n <= 32) {
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                C[i][j]=0;
                for (int k=0; k<n; k++) {
                    C[i][j] += A[i][k]*B[k][j];
                }
            }
        }
    }
    else {
        int half=n/2;

        int (*A11)[half], (*A12)[half], (*A21)[half], (*A22)[half];
        int (*B11)[half], (*B12)[half], (*B21)[half], (*B22)[half];
        int (*P1)[half], (*P2)[half], (*P3)[half], (*P4)[half], (*P5)[half],
            (*P6)[half], (*P7)[half];
        int (*S1)[half], (*S2)[half], (*S3)[half], (*S4)[half], (*S5)[half],
            (*S6)[half], (*S7)[half], (*S8)[half], (*S9)[half], (*S10)[half];
        
        A11=malloc(half*sizeof(*A11));
        A12=malloc(half*sizeof(*A12));
        A21=malloc(half*sizeof(*A21));
        A22=malloc(half*sizeof(*A22));
        B11=malloc(half*sizeof(*B11));
        B12=malloc(half*sizeof(*B12));
        B21=malloc(half*sizeof(*B21));
        B22=malloc(half*sizeof(*B22));
        P1=malloc(half*sizeof(*P1));
        P2=malloc(half*sizeof(*P2));
        P3=malloc(half*sizeof(*P3));
        P4=malloc(half*sizeof(*P4));
        P5=malloc(half*sizeof(*P5));
        P6=malloc(half*sizeof(*P6));
        P7=malloc(half*sizeof(*P7));
        S1=malloc(half*sizeof(*S1));
        S2=malloc(half*sizeof(*S2));
        S3=malloc(half*sizeof(*S3));
        S4=malloc(half*sizeof(*S4));
        S5=malloc(half*sizeof(*S5));
        S6=malloc(half*sizeof(*S6));
        S7=malloc(half*sizeof(*S7));
        S8=malloc(half*sizeof(*S8));
        S9=malloc(half*sizeof(*S9));
        S10=malloc(half*sizeof(*S10));

        // split A, B
        for (int i=0; i<half; i++) {
            for (int j=0; j<half; j++) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j+half];
                A21[i][j] = A[i+half][j];
                A22[i][j] = A[i+half][j+half];
              
                B11[i][j] = B[i][j];
                B12[i][j] = B[i][j+half];
                B21[i][j] = B[i+half][j];
                B22[i][j] = B[i+half][j+half];    
            }        
        }  
        
        matrix_sub(half, S1, B12, B22);         //S1=B12-B22
        matrix_add(half, S2, A11, A12);         //S2=A11+A12
        matrix_add(half, S3, A21, A22);         //S3=A21+A22
        matrix_sub(half, S4, B21, B11);         //S4=B21-B11
        matrix_add(half, S5, A11, A22);         //S5=A11+A22
        matrix_add(half, S6, B11, B22);         //S6=B11+B22
        matrix_sub(half, S7, A12, A22);         //S7=A12-A22
        matrix_add(half, S8, B21, B22);         //S8=B21+B22
        matrix_sub(half, S9, A11, A21);         //S9=A11-A21
        matrix_add(half, S10, B11, B12);        //S10=B11+B12

        free(A12);
        free(A21);
        free(B12);
        free(B21);

        #pragma omp parallel sections 
        {
            #pragma omp section 
            {
                Strassen_v2(half, P1, A11, S1); //P1=A11*S1
                free(A11);
                free(S1);
            }
            #pragma omp section 
            {
                Strassen_v2(half, P2, S2, B22); //P2=S2*B22
                free(S2);
                free(B22);
            }
            #pragma omp section 
            {
                Strassen_v2(half, P3, S3, B11); //P3=S3*B11
                free(S3);
                free(B11);
            }
            #pragma omp section 
            {
                Strassen_v2(half, P4, A22, S4); //P4=A22*S4
                free(A22);
                free(S4);
            }
            #pragma omp section 
            {
                Strassen_v2(half, P5, S5, S6);  //P5=S5*S6
                free(S5);
                free(S6);
            }
            #pragma omp section 
            {
                Strassen_v2(half, P6, S7, S8);  //P6=S7*S8
                free(S7);
                free(S8);
            }
            #pragma omp section 
            {
                Strassen_v2(half, P7, S9, S10); //P7=S9*S10
                free(S9);
                free(S10);
            }
        }
        
        int (*C11)[half], (*C12)[half], (*C21)[half], (*C22)[half],
            (*T1)[half], (*T2)[half];           //temporary array

        C11=malloc(half*sizeof(*C11));
        C12=malloc(half*sizeof(*C12));
        C21=malloc(half*sizeof(*C21));
        C22=malloc(half*sizeof(*C22));
        T1=malloc(half*sizeof(*T1));
        T2=malloc(half*sizeof(*T2));

        // C11=P5+P4-P2+P6
        matrix_add(half, T1, P5, P4);
        matrix_sub(half, T2, T1, P2);
        matrix_add(half, C11, T2, P6);
        
        // C12=P1+P2
        matrix_add(half, C12, P1, P2);
        
        // C21=P3+P4
        matrix_add(half, C21, P3, P4);
        
        // C22=P5+P1-P3-P7
        matrix_add(half, T1, P5, P1);
        matrix_sub(half, T2, T1, P3);
        matrix_sub(half, C22, T2, P7);

        free(P1);
        free(P2);
        free(P3);
        free(P4);
        free(P5);
        free(P6);
        free(P7);
        free(T1);
        free(T2);
        
        // combine C11, C12, C21, C22 into C
        for (int i=0; i<half; i++) {
            for (int j=0; j<half; j++) {
                C[i][j] = C11[i][j];
                C[i][j+half] = C12[i][j];
                C[i+half][j] = C21[i][j];
                C[i+half][j+half] = C22[i][j];        
            }        
        }

        free(C11);
        free(C12);
        free(C21);
        free(C22);

    }   //end else
}   //end strassen function




void matrix_add(int n, int C[][n], int A[][n], int B[][n])
{
    int i, j;
    for (i=0; i<n; i++)
        for (j=0; j<n; j++)
            C[i][j] = A[i][j]+B[i][j];
}

void matrix_sub(int n, int C[][n], int A[][n], int B[][n])
{
    int i, j;
    for (i=0; i<n; i++)
        for (j=0; j<n; j++)
            C[i][j] = A[i][j]-B[i][j];
}



void matrix_input(int n, int x[][n]) 
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
           scanf("%d", &x[i][j]);
        }        
    }
}

void matrix_output(int n, int x[][n])
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
           printf("%d ", x[i][j]);        
        }
        printf("\n");;        
    }     
}


void Strassen_v2(int n, int C[][n], int A[][n], int B[][n]) 
{
    if (n <= 32) {
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++){
                C[i][j]=0;
                for (int k=0; k<n; k++) {
                    C[i][j] += A[i][k]*B[k][j];
                }
            }
        }
    }
    else {
        int half=n/2;

        int (*A11)[half], (*A12)[half], (*A21)[half], (*A22)[half];
        int (*B11)[half], (*B12)[half], (*B21)[half], (*B22)[half];
        int (*P1)[half], (*P2)[half], (*P3)[half], (*P4)[half], (*P5)[half],
            (*P6)[half], (*P7)[half];
        int (*S1)[half], (*S2)[half], (*S3)[half], (*S4)[half], (*S5)[half],
            (*S6)[half], (*S7)[half], (*S8)[half], (*S9)[half], (*S10)[half];
        
        A11=malloc(half*sizeof(*A11));
        A12=malloc(half*sizeof(*A12));
        A21=malloc(half*sizeof(*A21));
        A22=malloc(half*sizeof(*A22));
        B11=malloc(half*sizeof(*B11));
        B12=malloc(half*sizeof(*B12));
        B21=malloc(half*sizeof(*B21));
        B22=malloc(half*sizeof(*B22));
        P1=malloc(half*sizeof(*P1));
        P2=malloc(half*sizeof(*P2));
        P3=malloc(half*sizeof(*P3));
        P4=malloc(half*sizeof(*P4));
        P5=malloc(half*sizeof(*P5));
        P6=malloc(half*sizeof(*P6));
        P7=malloc(half*sizeof(*P7));
        S1=malloc(half*sizeof(*S1));
        S2=malloc(half*sizeof(*S2));
        S3=malloc(half*sizeof(*S3));
        S4=malloc(half*sizeof(*S4));
        S5=malloc(half*sizeof(*S5));
        S6=malloc(half*sizeof(*S6));
        S7=malloc(half*sizeof(*S7));
        S8=malloc(half*sizeof(*S8));
        S9=malloc(half*sizeof(*S9));
        S10=malloc(half*sizeof(*S10));

        // split A, B
        for (int i=0; i<half; i++) {
            for (int j=0; j<half; j++) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j+half];
                A21[i][j] = A[i+half][j];
                A22[i][j] = A[i+half][j+half];
              
                B11[i][j] = B[i][j];
                B12[i][j] = B[i][j+half];
                B21[i][j] = B[i+half][j];
                B22[i][j] = B[i+half][j+half];    
            }        
        }  
        
        matrix_sub(half, S1, B12, B22);     //S1=B12-B22
        matrix_add(half, S2, A11, A12);     //S2=A11+A12
        matrix_add(half, S3, A21, A22);     //S3=A21+A22
        matrix_sub(half, S4, B21, B11);     //S4=B21-B11
        matrix_add(half, S5, A11, A22);     //S5=A11+A22
        matrix_add(half, S6, B11, B22);     //S6=B11+B22
        matrix_sub(half, S7, A12, A22);     //S7=A12-A22
        matrix_add(half, S8, B21, B22);     //S8=B21+B22
        matrix_sub(half, S9, A11, A21);     //S9=A11-A21
        matrix_add(half, S10, B11, B12);    //S10=B11+B12

        free(A12);
        free(A21);
        free(B12);
        free(B21);


        Strassen_v2(half, P1, A11, S1);     //P1=A11*S1
        free(A11);
        free(S1);
        Strassen_v2(half, P2, S2, B22);     //P2=S2*B22
        free(S2);
        free(B22);
        Strassen_v2(half, P3, S3, B11);     //P3=S3*B11
        free(S3);
        free(B11);
        Strassen_v2(half, P4, A22, S4);     //P4=A22*S4
        free(A22);
        free(S4);
        Strassen_v2(half, P5, S5, S6);      //P5=S5*S6
        free(S5);
        free(S6);
        Strassen_v2(half, P6, S7, S8);      //P6=S7*S8
        free(S7);
        free(S8);
        Strassen_v2(half, P7, S9, S10);     //P7=S9*S10
        free(S9);
        free(S10);

        
        int (*C11)[half], (*C12)[half], (*C21)[half], (*C22)[half],
            (*T1)[half], (*T2)[half];       //temporary array

        C11=malloc(half*sizeof(*C11));
        C12=malloc(half*sizeof(*C12));
        C21=malloc(half*sizeof(*C21));
        C22=malloc(half*sizeof(*C22));
        T1=malloc(half*sizeof(*T1));
        T2=malloc(half*sizeof(*T2));

        // C11=P5+P4-P2+P6
        matrix_add(half, T1, P5, P4);
        matrix_sub(half, T2, T1, P2);
        matrix_add(half, C11, T2, P6);
        
        // C12=P1+P2
        matrix_add(half, C12, P1, P2);
        
        // C21=P3+P4
        matrix_add(half, C21, P3, P4);
        
        // C22=P5+P1-P3-P7
        matrix_add(half, T1, P5, P1);
        matrix_sub(half, T2, T1, P3);
        matrix_sub(half, C22, T2, P7);

        free(P1);
        free(P2);
        free(P3);
        free(P4);
        free(P5);
        free(P6);
        free(P7);
        free(T1);
        free(T2);
        
        // combine C11, C12, C21, C22 into C
        for (int i=0; i<half; i++) {
            for (int j=0; j<half; j++) {
                C[i][j] = C11[i][j];
                C[i][j+half] = C12[i][j];
                C[i+half][j] = C21[i][j];
                C[i+half][j+half] = C22[i][j];        
            }        
        }

        free(C11);
        free(C12);
        free(C21);
        free(C22);

    }   //end else
}   //end strassen_v2 function
