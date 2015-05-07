CC = g++
CFLAGS = -lm -O2 -Wall -funroll-loops -ffast-math -pthread 
#CFLAGS = -lm -O2 -Wall

all: rnnlmlib.o rnnlm

rnnlmlib.o : rnnlmlib.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c rnnlmlib.cpp

rnnlm : rnnlm.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) rnnlm.cpp rnnlmlib.o -o rnnlm

clean:
	rm -rf *.o rnnlm
