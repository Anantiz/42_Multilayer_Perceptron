#include <iostream>
struct Test {
  typedef int foo;
};

template <typename T>
void f(typename T::foo) {std::cout<<"a"<<std::endl;}  // Definition #1

template <typename T>
void f(T) {std::cout<<"b"<<std::endl;}  // Definition #2

int main() {
  f<Test>(10);  // Call #1.
  f<int>(10);   // Call #2. Without error (even though there is no int::foo)
                // thanks to SFINAE.
  return 0;
}
