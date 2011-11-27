
TARGET = bin/recommender
#LIBS = -litpp_debug
LIBS = -litpp
DEFINES = -DALG_ITPP_IMPL -D_DEBUG
LIB_PATH = -L$(HOME)/develop.lib/itpp-4.2.0/lib
INCLUDES = -I inc -I $(HOME)/develop.lib/itpp-4.2.0/include -I $(HOME)/develop.lib/libkdtree-master/include -I $(HOME)/develop.lib/tclap/include

#CXX_OPT_FLAGS = -O0
CXX_OPT_FLAGS = -O3 -march=native

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++0x -pedantic -g -pipe $(CXX_OPT_FLAGS) $(INCLUDES) $(DEFINES)
LDFLAGS = $(LIB_PATH) $(LIBS)


all: prepare $(TARGET)

$(TARGET): obj/recommender.o src/error.hpp src/grouplens.hpp src/user_resemblance.hpp src/knn.hpp src/dataset_io.hpp src/cross_validation.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
obj/recommender.o: src/recommender.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


.PHONY: doc bin-package clean clean-all rebuild prepare
doc: 
	doxygen doc/Doxyfile
bin-package: clean-all all
	$(eval PACKAGE_NAME := $(TARGET)-0.0.1)
	$(eval PACKAGE_TMP_DIR := pkg/$(PACKAGE_NAME))
	-mkdir -p $(PACKAGE_TMP_DIR)
	cp $(TARGET) $(PACKAGE_TMP_DIR)
	-cp -r data $(PACKAGE_TMP_DIR)
	tar -cjvf $(PACKAGE_NAME).tar.bz2 -C pkg $(PACKAGE_NAME)
	rm -rf pkg
clean: 
	-rm -f core $(TARGET)
	-rm -f src/*.o src/*.gch inc/*.gch
	-rm -rf bin/* obj/*
clean-all: clean
	-rm -f *~ src/*~ inc/*~ doc/*~
	-rmdir bin obj
rebuild: clean all
prepare: 
	-mkdir -p obj bin
