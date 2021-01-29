#include "func.h"
//--------------------function--------------------//
// printUsage
void printUsage() {
    std::cout << "Check the input please."  << std::endl;
    std::cout << "a.out i inputFile t threshold" << std::endl;
    exit(0);
}
// checkOption
void checkOption(int argc, char **argv, Option &option) {
    if (argc%2 != 1) printUsage();  // parameter shoud be odd
    option.inputFile = "testData.fasta";  // input file name
    option.outputFile = "result.fasta";  // output file name
    option.threshold = 0.95;
    for (int i=1; i<argc; i+=2) {  // parsing parameters 
        switch (argv[i][0]) {
        case 'i':
            option.inputFile = argv[i+1];
            break;
        case 'o':
            option.outputFile = argv[i+1];
            break;
        case 't':
            option.threshold = std::atof(argv[i+1]);
            break;
        default:
            printUsage();
            break;
        }
    }
    if (option.threshold < 0.8 || option.threshold >= 1) {
        std::cout << "Threshold out of range." << std::endl;
        exit(0);
    }
    int temp = (option.threshold*100-80)/5;
    switch (temp) {  // wordLength decided by threshold
    case 0:  // threshold:0.80-0.85 wordLength:4
        option.wordLength = 4;
        break;
    case 1:  // threshold:0.85-0.90 wordLength:5
        option.wordLength = 5;
        break;
    case 2:  // threshold:0.90-0.95 wordLength:6
        option.wordLength = 6;
        break;
    case 3:  // threshold:0.90-1.00 wordLength:7
        option.wordLength = 7;
        break;
    }
    std::cout << "input:\t" << option.inputFile << std::endl;
    std::cout << "output:\t" << option.outputFile << std::endl;
    std::cout << "threshold:\t" << option.threshold << std::endl;
    std::cout << "word length:\t" << option.wordLength << std::endl;
}
// compare
bool compare(Read &a, Read &b) {
    return a.data.size() > b.data.size();
}
// readFile
void readFile(std::vector<Read> &reads, Option &option) {
    std::ifstream file(option.inputFile.c_str());
    Read read;
    std::string line;
    getline(file, line);
    read.name = line;
    while (getline(file, line)) {  // getline has no \n
        if (line[0] == '>') {
            reads.push_back(read);
            read.name = line;
            read.data = "";
            continue;
        }
        read.data += line;
    }
    reads.push_back(read);
    file.close();
    std::sort(reads.begin(), reads.end(), compare);
    std::cout << "reads count：\t" << reads.size() << std::endl;
}
// copyData
void copyData(std::vector<Read> &reads, Data &data) {
    Selector selector;
    sycl::queue queue(selector);
    std::cout << "Device:" <<
    queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    int readsCount = reads.size();
    data.readsCount = readsCount;
    data.lengths = sycl::malloc_shared<int>(readsCount, queue);
    data.offsets = sycl::malloc_shared<long>((readsCount + 1), queue);
    data.offsets[0] = 0;
    for (int i=0; i<readsCount; i++) {  // copy data for lengths and offsets
        int length = reads[i].data.size();
        data.lengths[i] = length;
        data.offsets[i+1] = data.offsets[i] + length/16*16+16;
    }
    data.reads = sycl::malloc_shared<char>(data.offsets[readsCount], queue);
    for (int i=0; i<readsCount; i++) {  // copy data for reads
        int offset = data.offsets[i];
        int length = data.lengths[i];
        memcpy(&data.reads[offset], reads[i].data.c_str(), length*sizeof(char));
    }
    queue.wait_and_throw();  // synchronize
}
// kernel_baseToNumber
void kernel_baseToNumber(char *reads, long length, sycl::nd_item<3> item) {
    long index = item.get_local_id(2) +
        item.get_local_range().get(2) * item.get_group(2);
    while (index < length) {
        switch (reads[index]) {
        case 'A':
            reads[index] = 0;
            break;
        case 'a':
            reads[index] = 0;
            break;
        case 'C':
            reads[index] = 1;
            break;
        case 'c':
            reads[index] = 1;
            break;
        case 'G':
            reads[index] = 2;
            break;
        case 'g':
            reads[index] = 2;
            break;
        case 'T':
            reads[index] = 3;
            break;
        case 't':
            reads[index] = 3;
            break;
        case 'U':
            reads[index] = 3;
            break;
        case 'u':
            reads[index] = 3;
            break;
        default:
            reads[index] = 4;
            break;
        }
        index += 128*128;
    }
}
// baseToNumber
void baseToNumber(Data &data) {
    Selector selector;
    sycl::queue queue(selector);
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];  // whole length
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>
        (sycl::range<3>(1, 1, 128) * sycl::range<3>(1, 1, 128),
        sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
            kernel_baseToNumber(data.reads, length, item);
        });
    });
    queue.wait_and_throw();  // synchronize
}
// 1 base use 2 bit，drop gap
// kernel_compressedData
void kernel_compressData(int *lengths, long *offsets, char *reads,
unsigned int *compressed, int *gaps, int readsCount, sycl::nd_item<3> item) {
    int index = item.get_local_id(2) +
    item.get_local_range().get(2) * item.get_group(2);
    if (index >= readsCount) return;  // out of range
    long mark = offsets[index]/16;  // compressed data offset
    int round = 0;  // write when round is 16
    int gapCount = 0;  // gap count
    unsigned int compressedTemp = 0;  // compressed data
    long start = offsets[index];
    long end = start + lengths[index];
    for (long i=start; i<end; i++) {
        unsigned char base = reads[i];  // read a base
        if (base < 4) {
            compressedTemp += base << (15-round)*2;
            round++;
            if (round == 16) {
                compressed[mark] = compressedTemp;
                compressedTemp = 0;
                round = 0;
                mark++;
            }
        } else {  // gap
            gapCount++;
        }
    }
    compressed[mark] = compressedTemp;
    gaps[index] = gapCount;
}
// compressData
void compressData(Data &data) {
    Selector selector;
    sycl::queue queue(selector);
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];
    data.compressed = sycl::malloc_shared<unsigned int>(length / 16, queue);
    data.gaps = sycl::malloc_shared<int>(readsCount, queue);
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
        (1, 1, (readsCount + 127) / 128) * sycl::range<3>(1, 1, 128),
        sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
            kernel_compressData(data.lengths, data.offsets,
            data.reads, data.compressed, data.gaps, readsCount, item);
        });
    });
    queue.wait_and_throw();  // synchronize
}
// kernel_createIndex4
void kernel_createIndex4(char *reads, int *lengths, long *offsets,
unsigned short *indexs, unsigned short *orders, long *words, int *magicBase,
int readsCount, sycl::nd_item<3> item) {
    int index = item.get_local_id(2) +
    item.get_local_range().get(2) * item.get_group(2);
    if (index >= readsCount) return;  // out of range
    int start = offsets[index];
    int end = start + lengths[index];
    int magic0=0, magic1=0, magic2=0, magic3=0;  // magic base
    char bases[4];
    for(int i=0; i<4; i++) {  // default is gap
        bases[i] = 4;
    }
    int wordCount = 0;
    for (int i=start; i<end; i++) {
        for(int j=0; j<3; j++) {  // copy base to array
            bases[j] = bases[j+1];
        }
        bases[3] = reads[i];
        switch (bases[3]) {  // update magic
            case 0:
                magic0++;
                break;
            case 1:
                magic1++;
                break;
            case 2:
                magic2++;
                break;
            case 3:
                magic3++;
                break;
        }
        unsigned short indexValue = 0;
        int flag = 0;  // if has gap
        for (int j=0; j<4; j++) {
            indexValue += (bases[j]&3)<<(3-j)*2;
            flag += sycl::max((int)(bases[j] - 3), 0);
        }
        indexs[i] = flag?65535:indexValue;  // gap value is 65535
        wordCount += flag?0:1;
    }
    words[index] = wordCount;  // index length
    magicBase[index*4+0] = magic0;  // update magicBase
    magicBase[index*4+1] = magic1;
    magicBase[index*4+2] = magic2;
    magicBase[index*4+3] = magic3;
}
// kernel_createIndex5
void kernel_createIndex5(char *reads, int *lengths, long *offsets,
unsigned short *indexs, unsigned short *orders, long *words, int *magicBase,
int readsCount, sycl::nd_item<3> item) {
    int index = item.get_local_id(2) +
    item.get_local_range().get(2) * item.get_group(2);
    if (index >= readsCount) return;
    int start = offsets[index];
    int end = start + lengths[index];
    int magic0=0, magic1=0, magic2=0, magic3=0;
    char bases[5];
    for(int i=0; i<5; i++) {
        bases[i] = 4;
    }
    int wordCount = 0;
    for (int i=start; i<end; i++) {
        for(int j=0; j<4; j++) {
            bases[j] = bases[j+1];
        }
        bases[4] = reads[i];
        switch (bases[4]) {
            case 0:
                magic0++;
                break;
            case 1:
                magic1++;
                break;
            case 2:
                magic2++;
                break;
            case 3:
                magic3++;
                break;
        }
        unsigned short indexValue = 0;
        int flag = 0;
        for (int j=0; j<5; j++) {
            indexValue += (bases[j]&3)<<(4-j)*2;
            flag += sycl::max((int)(bases[j] - 3), 0);
        }
        indexs[i] = flag?65535:indexValue;
        wordCount += flag?0:1;
    }
    words[index] = wordCount;
    magicBase[index*4+0] = magic0;
    magicBase[index*4+1] = magic1;
    magicBase[index*4+2] = magic2;
    magicBase[index*4+3] = magic3;
}
// kernel_createIndex6
void kernel_createIndex6(char *reads, int *lengths, long *offsets,
unsigned short *indexs, unsigned short *orders, long *words, int *magicBase,
int readsCount, sycl::nd_item<3> item) {
    int index = item.get_local_id(2) +
    item.get_local_range().get(2) * item.get_group(2);
    if (index >= readsCount) return;
    int start = offsets[index];
    int end = start + lengths[index];
    int magic0=0, magic1=0, magic2=0, magic3=0;
    char bases[6];
    for(int i=0; i<6; i++) {
        bases[i] = 4;
    }
    int wordCount = 0;
    for (int i=start; i<end; i++) {
        for(int j=0; j<5; j++) {
            bases[j] = bases[j+1];
        }
        bases[5] = reads[i];
        switch (bases[5]) {
            case 0:
                magic0++;
                break;
            case 1:
                magic1++;
                break;
            case 2:
                magic2++;
                break;
            case 3:
                magic3++;
                break;
        }
        unsigned short indexValue = 0;
        int flag = 0;
        for (int j=0; j<6; j++) {
            indexValue += (bases[j]&3)<<(5-j)*2;
            flag += sycl::max((int)(bases[j] - 3), 0);
        }
        indexs[i] = flag?65535:indexValue;
        wordCount += flag?0:1;
    }
    words[index] = wordCount;
    magicBase[index*4+0] = magic0;
    magicBase[index*4+1] = magic1;
    magicBase[index*4+2] = magic2;
    magicBase[index*4+3] = magic3;
}
// kernel_createIndex7
void kernel_createIndex7(char *reads, int *lengths, long *offsets,
unsigned short *indexs, unsigned short *orders, long *words, int *magicBase,
int readsCount, sycl::nd_item<3> item) {
    int index = item.get_local_id(2) +
                item.get_local_range().get(2) * item.get_group(2);
    if (index >= readsCount) return;
    int start = offsets[index];
    int end = start + lengths[index];
    int magic0=0, magic1=0, magic2=0, magic3=0;
    char bases[7];
    for(int i=0; i<7; i++) {
        bases[i] = 4;
    }
    int wordCount = 0;
    for (int i=start; i<end; i++) {
        for(int j=0; j<6; j++) {
            bases[j] = bases[j+1];
        }
        bases[6] = reads[i];
        switch (bases[6]) {
            case 0:
                magic0++;
                break;
            case 1:
                magic1++;
                break;
            case 2:
                magic2++;
                break;
            case 3:
                magic3++;
                break;
        }
        unsigned short indexValue = 0;
        int flag = 0;
        for (int j=0; j<7; j++) {
            indexValue += (bases[j]&3)<<(6-j)*2;
            flag += sycl::max((int)(bases[j] - 3), 0);
        }
        indexs[i] = flag?65535:indexValue;
        wordCount += flag?0:1;
    }
    words[index] = wordCount;
    magicBase[index*4+0] = magic0;
    magicBase[index*4+1] = magic1;
    magicBase[index*4+2] = magic2;
    magicBase[index*4+3] = magic3;
}
// createIndex
void createIndex(Data &data, Option &option) {
    Selector selector;
    sycl::queue queue(selector);
    int readsCount = data.readsCount;
    int wordLength = option.wordLength;
    int length = data.offsets[readsCount];
    data.indexs = sycl::malloc_shared<unsigned short>(length, queue); // index value
    data.orders = sycl::malloc_shared<unsigned short>(length, queue); // index rank
    data.words = sycl::malloc_shared<long>((readsCount + 1), queue); // index length
    data.magicBase = sycl::malloc_shared<int>(readsCount * 4, queue); // magic base
    switch (wordLength) {
        case 4:
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(
                sycl::range<3>(1, 1,
                (readsCount+127)/128)*sycl::range<3>(1, 1, 128),
                sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                    kernel_createIndex4(data.reads, data.lengths,
                    data.offsets, data.indexs, data.orders,
                    data.words, data.magicBase, readsCount, item);
                });
            });
            break;
        case 5:
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(
                sycl::range<3>(1, 1,
                (readsCount + 127) / 128) * sycl::range<3>(1, 1, 128),
                sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                    kernel_createIndex5(data.reads, data.lengths,
                    data.offsets,data.indexs, data.orders,
                    data.words, data.magicBase, readsCount, item);
                });
            });
            break;
        case 6:
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(
                sycl::range<3>(1, 1,
                (readsCount + 127) / 128) * sycl::range<3>(1, 1, 128),
                sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                    kernel_createIndex6(data.reads, data.lengths,
                    data.offsets, data.indexs, data.orders,
                    data.words, data.magicBase, readsCount, item);
                });
            });
            break;
        case 7:
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(
                sycl::range<3>(1, 1,
                (readsCount + 127) / 128) * sycl::range<3>(1, 1, 128),
                sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                    kernel_createIndex7(data.reads, data.lengths,
                    data.offsets, data.indexs, data.orders,
                    data.words, data.magicBase, readsCount, item);
                });
            });
            break;
    }
    queue.wait_and_throw();  // synchronize
}
// kernel_createCutoff
void kernel_createCutoff(float threshold, int wordLength,
int *lengths, long *words, int *wordCutoff, int readsCount,
sycl::nd_item<3> item) {
    int index = item.get_local_id(2) +
    item.get_group(2) * item.get_local_range().get(2);
    if (index >= readsCount) return;  // out of range
    int length = lengths[index];
    int required = length - wordLength + 1;
    int cutoff = sycl::ceil((float)length *
    (1.0 - threshold) * (float)wordLength);
    required -= cutoff;
    wordCutoff[index] = required;
}
// createCutoff
void createCutoff(Data &data, Option &option) {
    Selector selector;
    sycl::queue queue(selector);
    float threshold = option.threshold;
    int wordLength = option.wordLength;
    int readsCount = data.readsCount;
    data.wordCutoff = sycl::malloc_shared<int>(readsCount, queue); // word threshold
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>(
        sycl::range<3>(1, 1, (readsCount + 127) / 128) *
        sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
            kernel_createCutoff(threshold, wordLength, data.lengths,
            data.words, data.wordCutoff, readsCount, item);
        });
    });
    queue.wait_and_throw();  // synchronize
}
// sortIndex
void sortIndex(Data &data) {
    int readsCount = data.readsCount;
    for (int i=0; i<readsCount; i++) {
        int start = data.offsets[i];
        int length = data.words[i];
        std::sort(&data.indexs[start], &data.indexs[start]+length);
    }
}
// kernel_mergeIndex
void kernel_mergeIndex(long *offsets, unsigned short *indexs,
unsigned short *orders, long *words, int readsCount, sycl::nd_item<3> item) {
    int index = item.get_local_id(2) +
    item.get_local_range().get(2) * item.get_group(2);
    if (index >= readsCount) return;  // out of range
    int start = offsets[index];
    int end = start + words[index];
    unsigned short basePrevious = indexs[start];
    unsigned short baseNow;
    int count = 1;
    for (int i=start+1; i<end; i++) {  // merge same index，orders is count
        baseNow = indexs[i];
        if (baseNow == basePrevious) {
            count++;
            orders[i-1] = 0;
        } else {
            basePrevious = baseNow;
            orders[i-1] = count;
            count = 1;
        }
    }
    orders[end-1] = count;
}
// mergeIndex
void mergeIndex(Data &data) {
    Selector selector;
    sycl::queue queue(selector);
    int readsCount = data.readsCount;
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>(
        sycl::range<3>(1, 1, (readsCount + 127) / 128) *
        sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item) {
            kernel_mergeIndex(data.offsets, data.indexs, data.orders,
            data.words, readsCount, item);
        });
    });
    queue.wait_and_throw();  // synchronize
}
// updateRepresentative
void updateRepresentative(int *cluster, int &representative, int readsCount) {
    representative++;
    while (representative < readsCount) {
        if (cluster[representative] < 0) {  // is representative
            cluster[representative] = representative;
            break;
        }
        representative++;
    }
}
// kernel_makeTable
void kernel_makeTable(long *offsets, unsigned short *indexs,
unsigned short *orders, long *words,
unsigned short *table, int representative, sycl::nd_item<3> item) {
    int index = item.get_local_id(2) +
    item.get_local_range().get(2) * item.get_group(2);
    int start = offsets[representative];
    int end = start + words[representative];
    for (int i=index+start; i<end; i+=128*128) {
        unsigned short order = orders[i];
        if (order == 0) continue;
        table[indexs[i]] = order;
    }
}
// kernel_cleanTable
void kernel_cleanTable(long *offsets, unsigned short *indexs,
unsigned short *orders,  long *words,
unsigned short *table, int representative, sycl::nd_item<3> item) {
    int index = item.get_local_id(2) +
    item.get_local_range().get(2) * item.get_group(2);
    int start = offsets[representative];
    int end = start + words[representative];
    for (int i=index+start; i<end; i+=128*128) {
        unsigned short order = orders[i];
        if (order == 0) continue;
        table[indexs[i]] = 0;
    }
}
// kernel_magic
void kernel_magic(float threshold, int *lengths, int *magicBase,
int *cluster, int representative, int readsCount,
sycl::nd_item<3> item, const sycl::stream &out) {
    int index = item.get_local_id(2) +
    item.get_local_range().get(2) * item.get_group(2);
    if (index >= readsCount) return;  // out of range
    if (cluster[index] >= 0) return;  // is clustered
    int offsetOne = representative*4;  // representative magic offset
    int offsetTwo = index*4;  // query magic offset
    int magic = sycl::min(magicBase[offsetOne + 0], magicBase[offsetTwo + 0]) +
                sycl::min(magicBase[offsetOne + 1], magicBase[offsetTwo + 1]) +
                sycl::min(magicBase[offsetOne + 2], magicBase[offsetTwo + 2]) +
                sycl::min(magicBase[offsetOne + 3], magicBase[offsetTwo + 3]);
    int length = lengths[index];
    int minLength = ceil((float)length*threshold);
    if (magic > minLength) {  // pass filter
        cluster[index] = -2;
    }
}
// kernel_filter
void kernel_filter(float threshold, int wordLength, int *lengths,
long *offsets, unsigned short *indexs, unsigned short *orders, long *words,
int *wordCutoff, int *cluster, unsigned short *table, int readsCount,
sycl::nd_item<3> item, int *result, const sycl::stream &out) {
    if (item.get_group(2) >= readsCount) return;  // out of range
    if (cluster[item.get_group(2)] != -2) return; // out of filter
    result[item.get_local_id(2)] = 0;             // result in thread
    int start = offsets[item.get_group(2)];
    int end = start + words[item.get_group(2)];
    for (int i = item.get_local_id(2) + start; i < end; i += 128) {
        result[item.get_local_id(2)] += sycl::min(table[indexs[i]], orders[i]);
    }
    item.barrier();             // synchronize
    if (item.get_local_id(2) == 0) {
        for (int i=1; i<128; i++) {
            result[0] += result[i];
        }
    } else {
        return;
    }
    if (result[0] > wordCutoff[item.get_group(2)]) { // pass filter
        cluster[item.get_group(2)] = -3;
    } else {
        cluster[item.get_group(2)] = -1; // not pass filter
    }
}
// kernel_align
void kernel_align(float threshold, int *lengths, long *offsets, 
unsigned int *compressed, int *gaps, int representative,
int *cluster, int readsCount, sycl::nd_item<3> item, const sycl::stream &out) {
    // variables
    int index = item.get_local_id(2) +
                item.get_local_range().get(2) * item.get_group(2);
    if (index >= readsCount) return;  // out of range
    if (cluster[index] != -3) return;  // not pass filter
    int target = representative;  // representative read
    int query = index;  // query read
    int minLength = sycl::ceil((float)lengths[index] * threshold);
    int targetLength = lengths[target] - gaps[target];  // representative base count
    int queryLength = lengths[query] - gaps[query];  // query base count
    int target32Length = targetLength/16+1;  // compressed target length
    int query32Length  = queryLength/16+1;  // compressed query length
    int targetOffset = offsets[target]/16;  // representative offset
    int queryOffset = offsets[query]/16;  // query offset
    short rowNow[3000] = {0};  // dynamic matrix row
    short rowPrevious[3000] = {0};  // dynamic matrix row
    int columnPrevious[17] = {0};  // dynamic matrix column
    int columnNow[17] = {0};  // dynamic matrix column
    int shift = sycl::ceil((float)targetLength - (float)queryLength*threshold);
    shift = sycl::ceil((float)shift / 16.0); // shift form diagonal
    // compute
    for (int i = 0; i < query32Length; i++) {  // query is column
        // first big loop
        for (int j=0; j<17; j++) {
            columnPrevious[j] = 0;
            columnNow[j] = 0;
        }
        int targetIndex = 0;  // target position
        unsigned int queryPack = compressed[queryOffset+i];  // get 16 query bases
        int jstart = i-shift;
        jstart = sycl::max(jstart, 0);
        int jend = i+shift;
        jend = sycl::min(jend, target32Length);
        for (int j=0; j<target32Length; j++) {  // target is row
            columnPrevious[0] = rowPrevious[targetIndex];
            unsigned int targetPack = compressed[targetOffset+j];  // get 16 target bases
            //---16*16core----//
            for (int k=30; k>=0; k-=2) {  // 16 loops get target bases
                // first small loop
                int targetBase = (targetPack>>k)&3;  // get base from target
                int m=0;
                columnNow[m] = rowPrevious[targetIndex+1];
                for (int l=30; l>=0; l-=2) {  // 16 loops get query bases
                    m++;
                    int queryBase = (queryPack>>l)&3;  // get base from query
                    int diffScore = queryBase == targetBase;
                    columnNow[m] = columnPrevious[m-1] + diffScore;
                    columnNow[m] = sycl::max(columnNow[m], columnNow[m - 1]);
                    columnNow[m] = sycl::max(columnNow[m], columnPrevious[m]);
                }
                targetIndex++;
                rowNow[targetIndex] = columnNow[16];
                if (targetIndex == targetLength) {  // last column of dynamic matirx
                    if(i == query32Length-1) {  // complete
                        int score = columnNow[queryLength%16];
                        if (score >= minLength) {
                            cluster[index] = target;
                        } else {
                            cluster[index] = -1;
                        }
                        return;
                    }
                    break;
                }
                // secode small loop exchange columnPrevious and columnNow
                k-=2;
                targetBase = (targetPack>>k)&3;
                m=0;
                columnPrevious[m] = rowPrevious[targetIndex+1];
                for (int l=30; l>=0; l-=2) {
                    m++;
                    int queryBase = (queryPack>>l)&3;
                    int diffScore = queryBase == targetBase;
                    columnPrevious[m] = columnNow[m-1] + diffScore;
                    columnPrevious[m] =
                        sycl::max(columnPrevious[m], columnPrevious[m - 1]);
                    columnPrevious[m] =
                        sycl::max(columnPrevious[m], columnNow[m]);
                }
                targetIndex++;
                rowNow[targetIndex] = columnPrevious[16];
                if (targetIndex == targetLength) {
                    if(i == query32Length-1) {
                        int score = columnPrevious[queryLength%16];
                        if (score >= minLength) {
                            cluster[index] = target;
                        } else {
                            cluster[index] = -1;
                        }
                        return;
                    }
                    break;
                }
            }
        }
        // secode big loop exchage rowPrevious and rowNow
        i++;
        for (int j=0; j<17; j++) {
            columnPrevious[j] = 0;
            columnNow[j] = 0;
        }
        targetIndex = 0;
        queryPack = compressed[queryOffset+i];
        jstart = i-shift;
        jstart = sycl::max(jstart, 0);
        jend = i+shift;
        jend = sycl::min(jend, target32Length);
        for (int j=0; j<target32Length; j++) {
            unsigned int targetPack = compressed[targetOffset+j];
            //---16*16 core----//
            for (int k=30; k>=0; k-=2) {
                // first small loop
                int targetBase = (targetPack>>k)&3;
                int m=0;
                columnNow[m] = rowNow[targetIndex+1];
                for (int l=30; l>=0; l-=2) {
                    m++;
                    int queryBase = (queryPack>>l)&3;
                    int diffScore = queryBase == targetBase;
                    columnNow[m] = columnPrevious[m-1] + diffScore;
                    columnNow[m] = sycl::max(columnNow[m], columnNow[m - 1]);
                    columnNow[m] = sycl::max(columnNow[m], columnPrevious[m]);
                }
                targetIndex++;
                rowPrevious[targetIndex] = columnNow[16];
                if (targetIndex == targetLength) {
                    if(i == query32Length-1) {
                        int score = columnNow[queryLength%16];
                        if (score >= minLength) {
                            cluster[index] = target;
                        } else {
                            cluster[index] = -1;
                        }
                        return;
                    }
                    break;
                }
                // second small loop
                k-=2;
                targetBase = (targetPack>>k)&3;
                m=0;
                columnPrevious[m] = rowNow[targetIndex+1];
                for (int l=30; l>=0; l-=2) {
                    m++;
                    int queryBase = (queryPack>>l)&3;
                    int diffScore = queryBase == targetBase;
                    columnPrevious[m] = columnNow[m-1] + diffScore;
                    columnPrevious[m] =
                        sycl::max(columnPrevious[m], columnPrevious[m - 1]);
                    columnPrevious[m] =
                        sycl::max(columnPrevious[m], columnNow[m]);
                }
                targetIndex++;
                rowPrevious[targetIndex] = columnPrevious[16];
                if (targetIndex == targetLength) {
                    if(i == query32Length-1) {
                        int score = columnPrevious[queryLength%16];
                        if (score >= minLength) {
                            cluster[index] = target;
                        } else {
                            cluster[index] = -1;
                        }
                        return;
                    }
                    break;
                }
            }
        }
    }
}
// clustering
void clustering(Option &option, Data &data, Bench &bench) {
    Selector selector;
    sycl::queue queue(selector);
    float threshold = option.threshold;
    int wordLength = option.wordLength;
    int readsCount = data.readsCount;
    bench.cluster = sycl::malloc_shared<int>(readsCount, queue); // clustering result
    for (int i=0; i<readsCount; i++) {
        bench.cluster[i] = -1;
    }
    bench.table = sycl::malloc_shared<unsigned short>(65536, queue); // table
    memset(bench.table, 0, 65536*sizeof(unsigned short));  // fill zero
    bench.representative = -1;
    std::cout << "representative now / all reads count:" << std::endl;
    while (bench.representative < readsCount) {  // clustering
        updateRepresentative(bench.cluster, bench.representative, readsCount);  // update representative
        if (bench.representative >= readsCount-1) {  // complete
            return;
        }
        std::cout << bench.representative << "/" << readsCount << std::endl;
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128) *
            sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
            [=](sycl::nd_item<3> item) {
                kernel_makeTable(data.offsets, data.indexs, data.orders,
                data.words, bench.table, bench.representative, item);
            });
        }); // create table
        queue.submit([&](sycl::handler &cgh) {
            sycl::stream out(1024, 256, cgh);  // output
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
            (1, 1, (readsCount + 127) / 128) * sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                kernel_magic(threshold, data.lengths, data.magicBase,
                bench.cluster, bench.representative, readsCount, item, out);
            });
        }); // magic filter
        queue.submit([&](sycl::handler &cgh) {
            sycl::stream out(1024, 256, cgh);
            sycl::accessor<int, 1, sycl::access::mode::read_write,
            sycl::access::target::local>result_acc_ct1
            (sycl::range<1>(128), cgh);
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
            (1, 1, readsCount) * sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                kernel_filter(threshold, wordLength, data.lengths,
                data.offsets, data.indexs, data.orders, data.words,
                data.wordCutoff, bench.cluster, bench.table,
                readsCount, item, result_acc_ct1.get_pointer(), out);
            });
        }); // word filter
        queue.submit([&](sycl::handler &cgh) {
            sycl::stream out(1024, 256, cgh);
            cgh.parallel_for(sycl::nd_range<3>(
            sycl::range<3>(1, 1, (readsCount + 127) / 128) *
            sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
            [=](sycl::nd_item<3> item) {
                kernel_align(threshold, data.lengths, data.offsets,
                data.compressed, data.gaps, bench.representative,
                bench.cluster, readsCount, item, out);
            });
        }); // dynamic programming
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128) *
            sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
            [=](sycl::nd_item<3> item) {
                kernel_cleanTable(data.offsets, data.indexs, data.orders,
                data.words, bench.table, bench.representative, item);
            });
        }); // table fill zero
        queue.wait_and_throw();  // synchronize
    }
    queue.wait_and_throw();  // synchronize
}
// saveFile
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench) {
    std::ofstream file(option.outputFile.c_str());
    int sum = 0;
    for (int i=0; i<reads.size(); i++) {
        if (bench.cluster[i] == i) {
            file << reads[i].name << std::endl;
            file << reads[i].data << std::endl;
            sum++;
        }
    }
    file.close();
    std::cout << "cluster count：" << sum << std::endl;
}
