#include <stdio.h>
#include <math.h>

int is_prime(int n)
{
  int i;
  for (i=2;i<n;i++){
    if (n%i == 0){
      return false;
    }
  }
  return true;
}

// you just have to find out if the number is divided by numbers under root n
int is_prime_under_root(int n)
{
  int i, root;
  root = (int)sqrt(n);
  for (i=2;i<=root;i++){
    if (n%i==0){
      return false;
    }
  }
  return true;
}

// write main function below
