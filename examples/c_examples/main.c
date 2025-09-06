#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern double fibonacci(double a);

int myFib(int a) {
    if (a == 0) {
        return a;
    } else if (a == 1) {
        return a;
    } else {
        return myFib(a - 1) + myFib(a - 2);
    }

    return 0;
}

int main() {
    for (double i = 0; i < 20; i++) {
        if ((double)(myFib((int)i)) != fibonacci(i)) {
            printf("Oops, wrong fib values for integer %f\n", i);
            return 1;
        }
    }

    printf("Verification finished successfully.\n");
    return 0;
}