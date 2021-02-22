#include "func.h"
//--------------------主体--------------------//
int main(int argc, char **argv) {
    clock_t start;
    Option option;
    checkOption(argc, argv, option);  // 检查配置 ok
    std::vector<Read> reads;
    readFile(reads, option);  // 读文件 ok
    Data data;
    copyData(reads, data);  // 拷贝数据 ok
    baseToNumber(data);  // 碱基转数字 ok
    compressData(data);  // 压缩数据
    createIndex(data, option);  // 生成index ok
    createCutoff(data, option);  // 生成阈值 ok
    sortIndex(data);  // 排序index ok
    mergeIndex(data);  // 合并index ok

start=clock();
    Bench bench;
    clustering(option, data, bench);  // 聚类
std::cout << "聚类耗时：" << (clock()-start)/1000 << "ms" << std::endl;

    saveFile(option, reads, bench);
    checkValue(data);
    checkError();
}