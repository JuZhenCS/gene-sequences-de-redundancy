#pragma once
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream> // cout
#include <fstream>  // ifstream
#include <vector>  // vector
#include <cstring>  // memcpy
#include <algorithm>  // sort
//--------------------data--------------------//
struct Option {
    std::string inputFile;
    std::string outputFile;
    float threshold;
    int wordLength;
};
struct Read {
    std::string data;
    std::string name;
};
struct Data {
    // data form file
    int readsCount;
    int *lengths;
    long *offsets;
    char *reads;
    // data form program
    unsigned int *compressed;
    int *gaps;
    unsigned short *indexs;
    unsigned short *orders;
    long *words;
    int *magicBase;
    // threshold
    int *wordCutoff;
    int *baseCutoff;
};
struct Bench {
    int *cluster;
    unsigned short *table;
    int representative;
};
class Selector : public sycl::device_selector {
public:
    int operator()(const sycl::device &device) const override {
        int device_rating;
        if (device.is_gpu()) {  // gpu first
            device_rating = 4;
        } else if (device.get_info<cl::sycl::info::device::name>()
            == "Intel(R) Gen9 HD Graphics NEO") {  // Gen9 secode
            device_rating = 3;
        } else if (device.is_cpu()) {  // cpu last
            device_rating = 0;
        } else {  // xpus
            device_rating = 1;
        }
        return device_rating;
    };
};
//--------------------function--------------------//
void checkOption(int argc, char **argv, Option &option);
void readFile(std::vector<Read> &reads, Option &option);
void copyData(std::vector<Read> &reads, Data &data);
void baseToNumber(Data &data);
void compressData(Data &data);
void createIndex(Data &data, Option &option);
void createCutoff(Data &data, Option &option);
void sortIndex(Data &data);
void mergeIndex(Data &data);
void clustering(Option &option, Data &data, Bench &bench);
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench);
