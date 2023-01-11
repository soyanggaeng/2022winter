// Sieve of Eratosthenes

#include <stdio.h>
#include <stdlib.h> // atoi
#include <string.h> // memset

int main(int argc, char* argv[])
{
  if (argc<2){
    printf("Usage:prime2[integer]\n");
    return 0;
  }
  
  int n = atoi(argv[1]);
  if (n<2){
    printf("Error:n must be greater than 1\n");
    return 0;
  }
  
  int *parray;
  
  // memory allocation
  parray = new int [n+1];
  if (parray == 0){
    printf("Error: memory allocation failed\n");
    return 0;
  }
  
  // memory initialization
  memset(parray, 0, sizeof(int)*(n+1));
  
  int i,j;
  // prime number loop
  for (i=2; i<=n; i++){
    if (parray[i] == 1)
      continue;
    j=i;
    while ((j+=1)<=n){
      parray[j]=1; // i and its multiples not a prime number
    }
  }
  
  // print prime numbers
  for (i=2; i<=n; i++){
    if (parray[i]==0)
      printf("%d ", i);
  }
  printf("\n");
    
  // memory delete
  delete[] parray;
  return 0;
}
