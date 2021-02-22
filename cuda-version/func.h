// 基因序列去冗余应用
#include <iostream>  // cout
#include <fstream>  // ifstream
#include <ctime>  // clock
#include <vector>  // vector
#include <cstring>  // memcpy
#include <algorithm>  // sort
//--------------------数据--------------------//
struct Option {  // 配置选项
    std::string inputFile;  // 输入文件名
    std::string outputFile;  // 输出文件名
    float threshold;  // 阈值
    int wordLength;  // 词长度
};
struct Read {
    std::string data;
    std::string name;
};
struct Data {
    // 读入的数据
    int readsCount;  // 读长的数量
    int *lengths;  // 存储读长的长度
    long *offsets;  // 存储读长的开端
    char *reads;  // 存储读长
    // 生成的数据
    unsigned int *compressed;  // 压缩后的reads
    int *gaps;  // gap的数量
    unsigned short *indexs;  // 存储生成的index
    unsigned short *orders;  // 存储index的序号
    long *words;  // 存储序列的word数
    int *magicBase;  // 神奇碱基数
    // 阈值
    int *wordCutoff;  // word的阈值
    int *baseCutoff;  // 比对的阈值
};
struct Bench {  // workbench 工作台
    int *cluster;  // 存储聚类的结果
    unsigned short *table;  // 存储table
    int representative;  // 当前代表序列
};
//--------------------函数--------------------//
void checkOption(int argc, char **argv, Option &option);
void readFile(std::vector<Read> &reads, Option &option);
void copyData(std::vector<Read> &reads, Data &data);
void baseToNumber(Data &data);
void compressData(Data &data);
void createIndex(Data &data, Option &option);
void createCutoff(Data &data, Option &option);
void sortIndex(Data &data);
void mergeIndex(Data &data);
void updateRepresentative(int *cluster, int &representative, int readsCount);
void clustering(Option &option, Data &data, Bench &bench);
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench);
void checkValue(Data &data);
void checkError();
