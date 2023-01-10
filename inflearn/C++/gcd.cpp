// gcd: Greatest Common Divisor

#include <stdio.h>

int gcd_modulus(int a, int b)
{
  int k;
  while (b)
  {
    k = a % b;
    a = b;
    b = k;
  }
  return a;
}

int gcd_recursion(int a, b)
{
  if (b==0)
    return a;
  else
    return gcd_recursion(b, a%b);
}
 
// two ways all possible. recursion more sophisticated
