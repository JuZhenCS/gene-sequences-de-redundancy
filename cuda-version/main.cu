#include "func.h"
//--------------------main--------------------//
int main(int argc, char **argv) {
    clock_t start = clock();
    //----Data Preparation----//
    Option option;
    checkOption(argc, argv, option);  // check option ok
    std::vector<Read> reads;
    readFile(reads, option);  // read file ok
    Data data;
    copyData(reads, data);  // copy data ok
    //----pretreatment----//
    baseToNumber(data);  // base to number ok
    if (option.drop == 0 && option.pigeon == 1) {
        createPigeon(data);  // create pigeon ok
    }
    compressData(data);  // compress data ok
    createCutoff(data, option);  // create threshold ok
    if (option.drop == 0) {  // no dorp filter
        createPrefix(data);  // create prefilter ok
        createWords(data, option);  // create word ok
        sortWords(data, option);  // sort word ok
        mergeWords(data);  // merge word ok
    }
    //----remove redundancy----//
    Bench bench;
    if (option.drop == 0) {  // no dorp filter
        clustering(option, data, bench);  // with filer ok
    } else {
        clusteringDrop(option, data, bench);  // without filter ok
    }
    //----end----//
    saveFile(option, reads, bench);  // save result ok
    checkError();
    std::cout << "time consuming:\t" << (clock()-start)/1000 << "ms" << std::endl;
}
