#pragma once
#include <iostream>  // cout
#include <fstream>  // ifstream
#include <ctime>  // clock
#include <vector>  // vector
#include <cstring>  // memcpy
#include <algorithm>  // sort
//--------------------Data--------------------//
struct Option {  // option
    std::string inputFile;  // input file
    std::string outputFile;  // output file
    float threshold;  // threshold
    int wordLength;  // word length
    int drop;  // with or without filter
    int pigeon;  // with or without pigeon
};
struct Read {  // gene reads
    std::string name;
    std::string data;
};
struct Data {  // data in memory
    // data form file
    int readsCount;  // amount of reads
    int *lengths;  // length of reads
    long *offsets;  // offset of reads
    char *reads;  // reads
    // generated data
    unsigned int *compressed;  // compressed reads
    int *gaps;  // amoutn of gap
    int *prefix;  // prefilter
    unsigned int *words;  // word
    int *wordCounts;  // amount of words
    unsigned short *orders;  // order of words
    // threshold
    int *wordCutoff;  // threshold of word
    int *baseCutoff;  // threshold of align
    int *pigeonCutoff;  // threshold of pigeon
    // pigeon
    unsigned short *pigeonIndex;  // pigeon code
    unsigned short *pigeon;  // pigeon position
};
struct Bench {  // work bench
    unsigned short *table;  // table
    int *cluster;  // result of clustering
    int *remainList;  // unclustering
    int remainCount;  // amount of unclustering
    int *jobList;  // job list
    int jobCount;  // job count
    int representative;  // representative
};
//--------------------Function--------------------//
// checkOption check input
void checkOption(int argc, char **argv, Option &option);
// readFile read file
void readFile(std::vector<Read> &reads, Option &option);
// copyData copy data
void copyData(std::vector<Read> &reads, Data &data);
// baseToNumber convert base to number
void baseToNumber(Data &data);
// createPigeon generate pigeon
void createPigeon(Data &data);
// compressData compress data
void compressData(Data &data);
// createCutoff generate threshold
void createCutoff(Data &data, Option &option);
// createPrefix generate prefilter
void createPrefix(Data &data);
// createWords generate words
void createWords(Data &data, Option &option);
// sortWords sort words
void sortWords(Data &data, Option &option);
// mergeWords merge words
void mergeWords(Data &data);
// clustering clustering
void clustering(Option &option, Data &data, Bench &bench);
// clusteringDrop clustering without filter
void clusteringDrop(Option &option, Data &data, Bench &bench);

// saveFile save result
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench);
// checkValue check the result
void checkValue(Option &option, Data &data, Bench &bench);
// check GPU error
void checkError();
