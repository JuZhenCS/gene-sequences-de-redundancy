#include "func.h"
//--------------------main--------------------//
int main(int argc, char **argv) {
    Option option;
    checkOption(argc, argv, option);  // check configuration ok
    std::vector<Read> reads;
    readFile(reads, option);  // read file ok
    Data data;
    copyData(reads, data);  // copy data ok
    baseToNumber(data);  // base to number ok
    compressData(data);  // compress data ok
    createIndex(data, option);  // create index ok
    createCutoff(data, option);  // create threshold ok
    sortIndex(data);  // sort index ok
    mergeIndex(data);  // merge index ok
    Bench bench;
    clustering(option, data, bench);  // clustering ok
    saveFile(option, reads, bench);  // save result ok
}
