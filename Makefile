
TARGET = recommender
LIBS = -L$(HOME)/develop.lib/itpp-4.2.0/lib -litpp_debug
INCLUDES = -I inc -I $(HOME)/develop.lib/itpp-4.2.0/include -I $(HOME)/develop.lib/libkdtree-master/include
#DEBUG_DEFS = 
DEBUG_DEFS = -D_DEBUG

CXX = g++
CXXFLAGS = -Wall -std=c++0x -pedantic -O0 -g $(INCLUDES) $(DEBUG_DEFS)
LDFLAGS = $(LIBS)


all: prepare $(TARGET)

$(TARGET): obj/recommender.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
obj/recommender.o: src/recommender.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


.PHONY: bin-package clean clean-all rebuild prepare
doc:
	doxygen doc/Doxyfile
bin-package: clean all
	mkdir pkg
	cp $(TARGET) pkg
	cp -r data pkg
	tar -cjvf $(TARGET).tar.bz2 -C pkg/*
	rm -rf pkg
clean: 
	-rm -f core $(TARGET)
	-rm -f src/*.o src/*.gch inc/*.gch
	-rm -rf bin/* obj/*
clean-all: clean
	-rm -f *~ src/*~ inc/*~
	-rmdir bin obj
rebuild: clean all
prepare: 
	-mkdir -p obj bin
