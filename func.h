#ifndef FUNCH
#define FUNCH

#include <iostream> 
#include <fstream> 
#include <ctime> 
#include <vector> 
#include <cstring> 
#include <algorithm> 
#include <CL/sycl.hpp>

struct Option { 
    std::string inputFile; 
    std::string outputFile; 
    float threshold; 
    int wordLength; 
    int filter; 
    int pigeon; 
    int gpu; 
};
struct Read { 
    std::string name;
    std::string data;
};
struct Data { 
   
    int readsCount; 
    int *lengths; 
    long *offsets; 
    char *reads; 
   
    int *prefix; 
    unsigned short *words; 
    int *wordCounts; 
    unsigned short *orders; 
    int *gaps; 
    unsigned int *compressed; 
   
    int *wordCutoff; 
    int *baseCutoff; 
};
struct Bench { 
    unsigned short *table; 
    int *cluster; 
    int *remainList; 
    int remainCount; 
    int *jobList; 
    int jobCount; 
    int representative; 
};
void checkOption(int argc, char **argv, Option &option);
void readFile(std::vector<Read> &reads, Option &option);
void copyData(std::vector<Read> &reads, Data &data, Option &option);
void baseToNumber(Data &data);
void createPrefix(Data &data);
void createWords(Data &data, Option &option);
void sortWords(Data &data, Option &option);
void mergeWords(Data &data);
void createCutoff(Data &data, Option &option);
void deleteGap(Data &data);
void compressData(Data &data);
void clustering(Option &option, Data &data, Bench &bench);
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench);
#endif
