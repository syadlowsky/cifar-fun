CXX=g++
CPPFLAGS:= -O3 -fPIC -fopenmp -march=native -shared -std=c++0x -pedantic -Wall -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes -mavx -g3 -larmadillo
RM= rm -f
LDFLAGS= -shared
.PHONY: all clean


all: fastfood.so

fastfood.so: fastfood.o
	$(CXX) -shared $< -o $@

clean:
	rm -rf *.so
	rm -rf *.o



