#include "func.h"
sycl::device device; 
void printUsage() {
    std::cout << "use like this:"  << std::endl;
    std::cout << "a.out i inputFile t threshold" << std::endl;
    exit(0); 
}
void checkOption(int argc, char **argv, Option &option) {
    if (argc%2 != 1) printUsage(); 
    option.inputFile = "testData.fasta"; 
    option.outputFile = "result.fasta"; 
    option.threshold = 0.95; 
    option.wordLength = 0; 
    option.filter = 3; 
    option.pigeon = 0; 
    option.gpu = 1; 
    for (int i=1; i<argc; i+=2) { 
        switch (argv[i][0]) {
        case 'i':
            option.inputFile = argv[i+1];
            break;
        case 'o':
            option.outputFile = argv[i+1];
            break;
        case 't':
            option.threshold = std::stof(argv[i+1]);
            break;
        case 'w':
            option.wordLength = std::stoi(argv[i+1]);
            break;
        case 'f':
            option.filter = std::stoi(argv[i+1]);
            break;
        case 'p':
            option.pigeon = std::stoi(argv[i+1]);
            break;
        case 'g':
            option.gpu = std::stoi(argv[i+1]);
            break;
        default:
            printUsage();
            break;
        }
    }
    if (option.threshold < 0.8 || option.threshold >= 1) { 
        std::cout << "similarity error" << std::endl;
        std::cout << "0.8<=similarity<1" << std::endl;
        printUsage();
    }
    if (option.wordLength == 0) { 
        if (option.threshold<0.88) {
            option.wordLength = 4;
        } else if (option.threshold<0.94) {
            option.wordLength = 5;
        } else if (option.threshold<0.97) {
            option.wordLength = 6;
        } else {
            option.wordLength = 7;
        }
    } else {
        if (option.wordLength<4 || option.wordLength>8) {
            std::cout << "word length error" << std::endl;
            std::cout << "4<=word length<=8" << std::endl;
            printUsage();
        }
    }
    if (option.filter == 3) { 
        if (option.threshold <= 0.879) {
            option.filter = 0;
        } else {
            option.filter = 1;
        }
    } else {
        if (option.filter != 0 && option.filter != 1) {
            std::cout << "filer error" << std::endl;
            std::cout << "filter=0/1" << std::endl;
            printUsage();
        }
    }
    if (option.pigeon != 1 && option.pigeon != 0) { 
        std::cout << "pigeon error" << std::endl;
        std::cout << "pigeon=0/1" << std::endl;
        printUsage();
    }
    if (option.gpu != 1 && option.gpu != 0) { 
        std::cout << "gpu error" << std::endl;
        std::cout << "gpu=0/1" << std::endl;
        printUsage();
    }
    std::cout << "inputfile:\t" << option.inputFile << std::endl;
    std::cout << "outputfile:\t" << option.outputFile << std::endl;
    std::cout << "similarity:\t" << option.threshold << std::endl;
    std::cout << "word length:\t" << option.wordLength << std::endl;
    // std::cout << ":\t" << option.filter << std::endl;
}
void readFile(std::vector<Read> &reads, Option &option) {
    std::ifstream file(option.inputFile);
    Read read;
    std::string line;
    long end = 0; 
    long point = 0; 
    file.seekg(0, std::ios::end);
    end = file.tellg();
    file.seekg(0, std::ios::beg);
    while(true) {
        getline(file, line); 
        read.name = line;
        while (true) { 
            point = file.tellg();
            getline(file, line);
            if (line[0] == '>') { 
                file.seekg(point, std::ios::beg);
                reads.push_back(read);
                read.name = ""; 
                read.data = "";
                break;
            } else { 
                read.data += line;
            }
            point = file.tellg();
            if (point == end){ 
                reads.push_back(read);
                read.data = "";
                read.name = "";
                break;
            }
        }
        if (point == end) break; 
    }
    file.close();
    std::sort(reads.begin(), reads.end(), [](Read &a, Read &b) {
        return a.data.size() > b.data.size(); 
    }); 
    std::cout << "read file completed" << std::endl;
    std::cout << "shortest/longest:\t" << reads[0].data.size() << "/";
    std::cout << reads[reads.size()-1].data.size() << std::endl;
    std::cout << "reads:\t" << reads.size() << std::endl;
}
void copyData(std::vector<Read> &reads, Data &data, Option &option) {
    data.readsCount = reads.size();
    int readsCount = data.readsCount;
    if (option.gpu) {
        try { 
            device = sycl::device(sycl::gpu_selector());
        } catch (sycl::exception const& error) {
            std::cout << "Cannot select a GPU: " << error.what() << std::endl;
            std::cout << "Using a CPU device" << std::endl;
            try {
                device = sycl::device(sycl::cpu_selector());
            } catch (sycl::exception const& error) {
                std::cout << "no device" << std::endl;
                exit(0);
            }
        }
    } else {
        try { 
            device = sycl::device(sycl::cpu_selector());
        } catch (sycl::exception const& error) {
            std::cout << "Cannot select a CPU: " << error.what() << std::endl;
            std::cout << "Using a GPU device" << std::endl;
            try {
                device = sycl::device(sycl::gpu_selector());
            } catch (sycl::exception const& error) {
                std::cout << "no device" << std::endl;
                exit(0);
            }
        }
    }
    std::cout << "Using: " << device.get_info<sycl::info::device::name>();
    std::cout << std::endl;
    sycl::queue queue(device);
    data.lengths = sycl::malloc_shared<int>(readsCount, queue);
    data.offsets = sycl::malloc_shared<long>((readsCount+1), queue);
    data.offsets[0] = 0;
    for (int i=0; i<readsCount; i++) { 
        int length = reads[i].data.size();
        data.lengths[i] = length;
        data.offsets[i+1] = data.offsets[i]+length/32*32+32;
    }
    data.reads = sycl::malloc_shared<char>(data.offsets[readsCount], queue);
    for (int i=0; i<readsCount; i++) { 
        long start = data.offsets[i];
        int length = data.lengths[i];
        memcpy(&data.reads[start], reads[i].data.c_str(), length*sizeof(char));
    }
    queue.wait_and_throw(); 
    std::cout << "copy data completed" << std::endl;
}
void kernel_baseToNumber(char *reads, long length, sycl::nd_item<3> item) {
    long index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
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
void baseToNumber(Data &data) {
    int readsCount = data.readsCount;
    sycl::queue queue(device);
    try { 
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128)*
            sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
            [=](sycl::nd_item<3> item) {
                kernel_baseToNumber(data.reads, data.offsets[readsCount], item);
            });
        });
    } catch (sycl::exception const& error) {
        std::cout << "base to number:" << error.what() << std::endl;
    }
    queue.wait_and_throw(); 
    std::cout << "base to number completed" << std::endl;
}
void kernel_createPrefix(int *lengths, long *offsets, char *reads,
int *prefix, int readsCount, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= readsCount) return; 
    int base[5] = {0}; 
    long start = offsets[index]; 
    int length = lengths[index];
    for (int i=0; i<length; i++) {
        switch(reads[start+i]) {
            case 0:
                base[0] += 1;
                break;
            case 1:
                base[1] += 1;
                break;
            case 2:
                base[2] += 1;
                break;
            case 3:
                base[3] += 1;
                break;
            case 4:
                base[4] += 1;
                break;
        }
    }
    prefix[index*4+0] = base[0];
    prefix[index*4+1] = base[1];
    prefix[index*4+2] = base[2];
    prefix[index*4+3] = base[3];
}
void createPrefix(Data &data) {
    sycl::queue queue(device);
    int readsCount = data.readsCount;
    data.prefix = sycl::malloc_shared<int>(readsCount * 4, queue);
    try {
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
            (1, 1, (readsCount+127)/128)*sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                kernel_createPrefix(data.lengths, data.offsets, data.reads,
                data.prefix, readsCount, item);
            });
        });
    } catch (sycl::exception const& error) {
        std::cout << "make prefilter:" << error.what() << std::endl;
    }
    queue.wait_and_throw(); 
    std::cout << "make prefilter complelted" << std::endl;
}
void kernel_createWords(int *lengths, long *offsets, char *reads,
unsigned short *words, int *wordCounts, int readsCount, int wordLength,
sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= readsCount) return; 
    long start = offsets[index]; 
    int length = lengths[index];
    if (length < wordLength) { 
        wordCounts[index] = 0;
        return;
    }
    int count = 0; 
    for (int i=wordLength-1; i<length; i++) {
        unsigned short word = 0;
        int flag = 0; 
        for (int j=0; j<wordLength; j++) {
            unsigned char base = reads[start+i-j];
            word += base<<j*2;
            if (base == 4) flag = 1;
        }
        if (flag == 0) { 
            words[start+count] = word;
            count += 1;
        }
    }
    wordCounts[index] = count;
}
void createWords(Data &data, Option &option) {
    sycl::queue queue(device);
    int readsCount = data.readsCount;
    int wordLength = option.wordLength;
    int length = data.offsets[readsCount];
    data.words = sycl::malloc_shared<unsigned short>(length, queue);
    data.wordCounts = sycl::malloc_shared<int>(readsCount, queue);  
    queue.wait_and_throw();
    try { 
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
            (1, 1, (readsCount+127)/128)*sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                kernel_createWords(data.lengths, data.offsets, data.reads,
                data.words, data.wordCounts, readsCount, wordLength, item);
            });
        });
    } catch (sycl::exception const& error) {
        std::cout << "make word:" << error.what() << std::endl;
    }
    queue.wait_and_throw();
    std::cout << "make word completed" << std::endl;
}
void kernel_sortWords(long *offsets, unsigned short *words, int *wordCounts,
int wordLength, int readsCount, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= readsCount) return; 
    long start = offsets[index];
    int wordCount = wordCounts[index];
    for (int gap=wordCount/2; gap>0; gap/=2){ 
        for (int i=gap; i<wordCount; i++) {
            for (int j=i-gap; j>=0; j-=gap) {
                if (words[start+j] > words[start+j+gap]) {
                    unsigned int temp = words[start+j];
                    words[start+j] = words[start+j+gap];
                    words[start+j+gap] = temp;
                } else {
                    break;
                }
            }
        }
    }
}
void sortWords(Data &data, Option &option) {
    int readsCount = data.readsCount;
    sycl::queue queue(device);
    int wordLength = option.wordLength;
    try { 
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
            (1, 1, (readsCount+127)/128)*sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                kernel_sortWords(data.offsets, data.words, data.wordCounts,
                wordLength, readsCount, item);
            });
        });
    } catch (sycl::exception const& error) {
        std::cout << "sort word:" << error.what() << std::endl;
    }
    queue.wait_and_throw(); 
    std::cout << "sort word completed" << std::endl;
}
void kernel_mergeWords(long *offsets, unsigned short *words, int *wordCounts,
unsigned short *orders, int readsCount, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= readsCount) return; 
    long start = offsets[index];
    int wordCount = wordCounts[index];
    unsigned int preWord = words[start];
    unsigned int current;
    unsigned short count = 0;
    for (int i=0; i<wordCount; i++) { 
        current = words[start+i];
        if (preWord == current) {
            count += 1;
            orders[start+i] = 0;
        } else {
            preWord = current;
            orders[start+i] = 0;
            orders[start+i-1] = count;
            count = 1;
        }
    }
    orders[start+wordCount-1] = count;
}
void mergeWords(Data &data) {
    sycl::queue queue(device);
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];
    data.orders = sycl::malloc_shared<unsigned short>(length, queue);
    try { 
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
            (1, 1, (readsCount+127)/128)*sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                kernel_mergeWords(data.offsets, data.words, data.wordCounts,
                data.orders, readsCount, item);
            });
        });
    } catch (sycl::exception const& error) {
        std::cout << "merge word" << error.what() << std::endl;
    }
    queue.wait_and_throw();
    std::cout << "merge word completed" << std::endl;
}
void kernel_createCutoff(int *lengths, int *wordCutoff, int *baseCutoff,
float threshold, int wordLength, int readsCount, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_group(2)*item.get_local_range().get(2);
    if (index >= readsCount) return; 
   
    int length = lengths[index];
    int required = length - wordLength + 1;
    int cutoff = sycl::ceil((float)length*(1.0f-threshold))*wordLength;
    required -= cutoff;
    required = sycl::max(required, 1);
    float offset = 0; 
    if (threshold >= 0.9) {
        offset = 1.1 - sycl::fabs(threshold - 0.95) * 2;
    } else {
        offset = 1;
    }
    offset = 1;
    required = sycl::ceil((float)required * offset);
    wordCutoff[index] = required;
   
    required = sycl::ceil((float)length * threshold);
    baseCutoff[index] = required;
}
void createCutoff(Data &data, Option &option) {
    sycl::queue queue(device);
    int readsCount = data.readsCount;
    data.wordCutoff = sycl::malloc_shared<int>(readsCount, queue);
    data.baseCutoff = sycl::malloc_shared<int>(readsCount, queue);
    float threshold = option.threshold;
    int wordLength = option.wordLength;
    try { 
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
            (1, 1, (readsCount+127)/128)*sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                kernel_createCutoff(data.lengths, data.wordCutoff,
                data.baseCutoff, threshold,wordLength, readsCount, item);
            });
        });
    } catch (sycl::exception const& error) {
        std::cout << "make threshold:" << error.what() << std::endl;
    }
    queue.wait_and_throw(); 
    std::cout << "make threshold completed" << std::endl;
}
void kernel_deleteGap(int *lengths, long *offsets, char *reads,
int *gaps, int readsCount, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= readsCount) return; 
    long start = offsets[index];
    int length = lengths[index];
    int count = 0;
    int gap = 0;
    for (int i=0; i<length; i++) {
        char base = reads[start+i];
        if (base < 4) { 
            reads[start+count] = base;
            count += 1;
        } else { 
            gap += 1;
        }
    }
    gaps[index] = gap;
}
void deleteGap(Data &data) {
    sycl::queue queue(device);
    int readsCount = data.readsCount;
    data.gaps = sycl::malloc_shared<int>(readsCount, queue);
    try { 
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
            (1, 1, (readsCount+127)/128)*sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                kernel_deleteGap(data.lengths, data.offsets, data.reads,
                data.gaps, readsCount, item);
            });
        });
    } catch (sycl::exception const& error) {
        std::cout << "delete gap:" << error.what() << std::endl;
    }
    queue.wait_and_throw(); 
    std::cout << "delete gap completed" << std::endl;
}
void kernel_compressData(int *lengths, long *offsets, char *reads, int *gaps,
unsigned int *compressed, int readsCount, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= readsCount) return; 
    long readStart = offsets[index];
    long compressStart = readStart/16;
    int length = lengths[index] - gaps[index];
    length = length/32+1;
    for (int i=0; i<length; i++) {
        unsigned int low = 0;
        unsigned int hight = 0;
        for (int j=0; j<32; j++) {
            char base = reads[readStart+i*32+j];
            switch (base) {
                case 1:
                    low += 1<<j;
                    break;
                case 2:
                    hight += 1<<j;
                    break;
                case 3:
                    low += 1<<j;
                    hight += 1<<j;
                    break;
                default:
                    break;
            }
        }
        compressed[compressStart+i*2+0] = low;
        compressed[compressStart+i*2+1] = hight;
    }
}
void compressData(Data &data) {
    sycl::queue queue(device);
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];
    data.compressed = sycl::malloc_shared<unsigned int>(length/16, queue);
    try {
        queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
            (1, 1, (readsCount+127)/128)*sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                kernel_compressData(data.lengths, data.offsets, data.reads,
                data.gaps, data.compressed, readsCount, item);
            });
        });
    } catch (sycl::exception const& error) {
        std::cout << "compress:" << error.what() << std::endl;
    }
    queue.wait_and_throw(); 
    std::cout << "compress completed" << std::endl;
}
void initBench(Bench &bench, int readsCount) {
    sycl::queue queue(device);
    bench.table = sycl::malloc_shared<unsigned short>((1 << 2*8), queue);
    memset(bench.table, 0, (1<<2*8)*sizeof(unsigned short)); 
    bench.cluster = sycl::malloc_shared<int>(readsCount, queue);
    for (int i=0; i<readsCount; i++) { 
        bench.cluster[i] = -1;
    }
    bench.remainList = sycl::malloc_shared<int>(readsCount, queue);
    for (int i=0; i<readsCount; i++) { 
        bench.remainList[i] = i;
    }
    bench.remainCount = readsCount; 
    bench.jobList = sycl::malloc_shared<int>(readsCount, queue);
    for (int i=0; i<readsCount; i++) { 
        bench.jobList[i] = i;
    }
    bench.jobCount = readsCount; 
    bench.representative = -1;
    queue.wait_and_throw(); 
}
void updateRepresentative(int *cluster, int &representative, int readsCount) {
    representative += 1;
    while (representative < readsCount) {
        if (cluster[representative] == -1) { 
            cluster[representative] = representative;
            break;
        } else {
            representative += 1;
        }
    }
}
void updateRemain(int *cluster, int *remainList, int &remainCount) {
    int count = 0;
    for (int i=0; i<remainCount; i++) {
        int index = remainList[i];
        if (cluster[index] == -1) {
            remainList[count] = index;
            count += 1;
        }
    }
    remainCount = count;
}
void updatJobs(int *jobList, int &jobCount) {
    int count = 0;
    for (int i=0; i<jobCount; i++) {
        int value = jobList[i];
        if (value >= 0) {
            jobList[count] = value;
            count += 1;
        }
    }
    jobCount = count;
}
void kernel_makeTable(long *offsets, unsigned short *words,
int *wordCounts, unsigned short *orders,unsigned short *table,
int representative, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    long start = offsets[representative];
    int length = wordCounts[representative];
    for (int i=index; i<length; i+=128*128) {
        unsigned short word = words[start+i];
        unsigned short order = orders[start+i];
        if (order > 0) table[word] = order; 
    }
}
void kernel_cleanTable(long *offsets, unsigned short *words,
int* wordCounts, unsigned short *orders, unsigned short *table,
int representative, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    long start = offsets[representative];
    int length = wordCounts[representative];
    for (int i=index; i<length; i+=128*128) {
        unsigned short word = words[start+i];
        unsigned short order = orders[start+i];
        if (order > 0) table[word] = 0; 
    }
}
void kernel_preFilter(int *prefix, int *baseCutoff, int *jobList,
int jobCount, int representative, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= jobCount) return; 
    int text = representative; 
    int query = jobList[index]; 
    int offsetOne = text*4; 
    int offsetTwo = query*4; 
    int sum = 0;
    sum += sycl::min(prefix[offsetOne + 0], prefix[offsetTwo + 0]);
    sum += sycl::min(prefix[offsetOne + 1], prefix[offsetTwo + 1]);
    sum += sycl::min(prefix[offsetOne + 2], prefix[offsetTwo + 2]);
    sum += sycl::min(prefix[offsetOne + 3], prefix[offsetTwo + 3]);
    int cutoff = baseCutoff[query];
    if (sum < cutoff) { 
        jobList[index] = -1;
    }
}
void kernel_filter(long *offsets, unsigned short *words, int *wordCounts,
unsigned short *orders, int *wordCutoff, unsigned short *table, int *jobList,
int jobCount, sycl::nd_item<3> item, int *result) {
    if (item.get_group(2) >= jobCount) return; 
    int query = jobList[item.get_group(2)]; 
    result[item.get_local_id(2)] = 0; 
    long start = offsets[query];
    int length = wordCounts[query];
    for (int i = item.get_local_id(2); i<length; i+=128) {
        unsigned short value = words[start+i]; 
        result[item.get_local_id(2)] +=
        sycl::min(table[value], orders[start + i]);
    }
    item.barrier(sycl::access::fence_space::local_space);
    for (int i=128/2; i>0; i/=2) { 
        if (item.get_local_id(2) >= i) { 
        } else {
            result[item.get_local_id(2)] += result[item.get_local_id(2)+i];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if (item.get_local_id(2) == 0) {
        int cutoff = wordCutoff[query];
        if (result[0] < cutoff) { 
            jobList[item.get_group(2)] = -1;
        }
    }
}
void kernel_dynamic(int *lengths, long *offsets, int *gaps,
unsigned int *compressed, int *baseCutoff, int *cluster, int *jobList,
int jobCount, int representative, sycl::nd_item<3> item,
unsigned int *bases) {
   
    int text = representative;
    long textStart = offsets[text]/16;
    int textLength = lengths[text]-gaps[text];
    for (int i = item.get_local_id(2); i < textLength/32+1;
    i += item.get_local_range().get(2)) {
        bases[i*2+0] = compressed[textStart+i*2+0];
        bases[i*2+1] = compressed[textStart+i*2+1];
    }
   
   
    int index;
    index = item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= jobCount) return; 
    unsigned int line[2048] = {0xFFFFFFFF}; 
    for (int i=0; i<2048; i++) {
        line[i] = 0xFFFFFFFF;
    }
    int query = jobList[index];
    long queryStart = offsets[query] / 16;
    int queryLength = lengths[query] - gaps[query];
    for (int i=0; i<queryLength/32; i++) { 
        unsigned int column[32] = {0};
        unsigned int queryLow = compressed[queryStart+i*2+0];
        unsigned int queryHight = compressed[queryStart+i*2+1];
        for (int j=0; j<textLength/32+1; j++) { 
            unsigned int textl = bases[j*2+0];
            unsigned int texth = bases[j*2+1];
            unsigned int row = line[j];
            for (int k=0; k<32; k++) { 
                unsigned int queryl = 0x00000000;
                if (queryLow>>k&1) queryl = 0xFFFFFFFF;
                unsigned int queryh = 0x00000000;
                if (queryHight>>k&1) queryh = 0xFFFFFFFF;
                unsigned int temp1 = textl ^ queryl;
                unsigned int temp2 = texth ^ queryh;
                unsigned int match = (~temp1)&(~temp2);
                unsigned int unmatch = ~match;
                unsigned int temp3 = row & match;
                unsigned int temp4 = row & unmatch;
                unsigned int carry = column[k];
                unsigned int temp5 = row + carry;
                unsigned int carry1 = temp5 < row;
                temp5 += temp3;
                unsigned int carry2 = temp5 < temp3;
                carry = carry1 | carry2;
                row = temp5 | temp4;
                column[k] = carry; 
            }
            line[j] = row;
        }
    }
   
    unsigned int column[32] = {0};
    unsigned int queryLow = compressed[queryStart+(queryLength/32)*2+0];
    unsigned int queryHight = compressed[queryStart+(queryLength/32)*2+1];
    for (int j=0; j<textLength/32+1; j++) { 
        unsigned int textl = bases[j*2+0];
        unsigned int texth = bases[j*2+1];
        unsigned int row = line[j];
        for (int k=0; k<queryLength%32; k++) { 
            unsigned int queryl = 0x00000000;
            if (queryLow>>k&1) queryl = 0xFFFFFFFF;
            unsigned int queryh = 0x00000000;
            if (queryHight>>k&1) queryh = 0xFFFFFFFF;
            unsigned int temp1 = textl ^ queryl;
            unsigned int temp2 = texth ^ queryh;
            unsigned int match = (~temp1)&(~temp2);
            unsigned int unmatch = ~match;
            unsigned int temp3 = row & match;
            unsigned int temp4 = row & unmatch;
            unsigned int carry = column[k];
            unsigned int temp5 = row + carry;
            unsigned int carry1 = temp5 < row;
            temp5 += temp3;
            unsigned int carry2 = temp5 < temp3;
            carry = carry1 | carry2;
            row = temp5 | temp4;
            column[k] = carry;
        }
        line[j] = row;
    }
   
    int sum = 0;
    unsigned int result;
    for (int i=0; i<textLength/32; i++) {
        result = line[i];
        for (int j=0; j<32; j++) {
            sum += result>>j&1^1;
        }
    }
    result = line[textLength/32];
    for (int i=0; i<textLength%32; i++) {
        sum += result>>i&1^1;
    }
    int cutoff = baseCutoff[query];
    if (sum > cutoff) {
        cluster[query] = text;
    } else {
        jobList[index] = -1;
    }
}
void clustering(Option &option, Data &data, Bench &bench) {
    sycl::queue queue(device);
    int readsCount = data.readsCount;
    initBench(bench, readsCount); 
    std::cout << "now/whole:" << std::endl;
    while (true) { 
        updateRepresentative(bench.cluster, bench.representative, readsCount);
        if (bench.representative >= readsCount) break; 
       
        std::cout << "\r" << bench.representative+1 << "/" << readsCount;
        std::flush(std::cout); 
       
        updateRemain(bench.cluster, bench.remainList, bench.remainCount);
       
        memcpy(bench.jobList, bench.remainList, bench.remainCount*sizeof(int));
        bench.jobCount = bench.remainCount;
        queue.wait_and_throw(); 
       
        try { 
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128)*
                sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
                [=](sycl::nd_item<3> item) {
                    kernel_makeTable(data.offsets, data.words, data.wordCounts,
                    data.orders, bench.table, bench.representative, item);
                });
            });
        } catch (sycl::exception const& error) {
            std::cout << "make table:" << error.what() << std::endl;
        }
        queue.wait_and_throw(); 
        updatJobs(bench.jobList, bench.jobCount); 
        if (bench.jobCount > 0) {
            try { 
                queue.submit([&](sycl::handler &cgh) {
                    cgh.parallel_for(sycl::nd_range<3>
                    (sycl::range<3>(1, 1, (bench.jobCount+127)/128)*
                    sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
                    [=](sycl::nd_item<3> item) {
                        kernel_preFilter(data.prefix, data.baseCutoff,
                        bench.jobList, bench.jobCount,
                        bench.representative, item);
                    });
                });
            } catch (sycl::exception const& error) {
                std::cout << "prefilter:" << error.what() << std::endl;
            }
        }
        queue.wait_and_throw(); 
        updatJobs(bench.jobList, bench.jobCount); 
        if (bench.jobCount > 0) {
            try { 
            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor<int, 1, sycl::access::mode::read_write,
                sycl::access::target::local>result(sycl::range<1>(128), cgh);
                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
                (1, 1, bench.jobCount)*sycl::range<3>(1, 1, 128),sycl::range<3>
                (1, 1, 128)), [=](sycl::nd_item<3> item) {
                    kernel_filter(data.offsets, data.words, data.wordCounts,
                    data.orders, data.wordCutoff, bench.table, bench.jobList,
                    bench.jobCount, item, result.get_pointer());
                });
            });
            } catch (sycl::exception const& error) {
                std::cout << "word fiter:" << error.what() << std::endl;
            }
        }
        queue.wait_and_throw(); 
       
        updatJobs(bench.jobList, bench.jobCount); 
        if (bench.jobCount > 0) { 
            try { 
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor<unsigned int, 1,
                    sycl::access::mode::read_write, sycl::access::target::local>
                    bases_acc_ct1(sycl::range<1>(2048), cgh);
                    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>
                    (1, 1, (bench.jobCount+127)/128)*sycl::range<3>(1, 1, 128),
                    sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                        kernel_dynamic(data.lengths, data.offsets, data.gaps,
                        data.compressed, data.baseCutoff, bench.cluster,
                        bench.jobList, bench.jobCount, bench.representative,
                        item, bases_acc_ct1.get_pointer());
                    });
                });
            } catch (sycl::exception const& error) {
                std::cout << "dynamic: " << error.what() << std::endl;
            }
        }
       
        try { 
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128)*
                sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
                [=](sycl::nd_item<3> item) {
                    kernel_cleanTable(data.offsets, data.words, data.wordCounts,
                    data.orders, bench.table, bench.representative, item);
                });
            });
        } catch (sycl::exception const& error) {
            std::cout << "clean table:" << error.what() << std::endl;
        }
        queue.wait_and_throw(); 
    }
    std::cout << std::endl;
}
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench) {
    std::ofstream file(option.outputFile);
    int sum = 0;
    for (int i=0; i<reads.size(); i++) {
        if (bench.cluster[i] == i) {
            file << reads[i].name << std::endl;
            file << reads[i].data << std::endl;
            sum++;
        }
    }
    file.close();
    std::cout << "cluster:" << sum << std::endl;
}
