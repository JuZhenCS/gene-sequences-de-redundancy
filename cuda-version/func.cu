#include "func.h"


void printUsage() {
    std::cout << "Please check parameters."  << std::endl;
    std::cout << "a.out i inputFile t threshold" << std::endl;
    exit(0);
}

void checkOption(int argc, char **argv, Option &option) {
    if (argc%2 != 1) printUsage();
    option.inputFile = "testData.fasta";
    option.outputFile = "result.fasta";
    option.threshold = 0.95;
    option.wordLength = 0;
    option.drop = 3;
    option.pigeon = 0;
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
        case 'd':
            option.drop = std::stoi(argv[i+1]);
            break;
        case 'p':
            option.pigeon = std::stoi(argv[i+1]);
            break;
        default:
            printUsage();
            break;
        }
    }
    if (option.threshold < 0.8 || option.threshold >= 1) {  
        std::cout << "threshold out of range" << std::endl;
        std::cout << "0.8<=threshold<1" << std::endl;
        printUsage();
        exit(0);
    }
    if (option.wordLength == 0) {  
        if (option.threshold<0.88) {
            option.wordLength = 4;
        } else if (option.threshold<0.94) {
            option.wordLength = 7;
        } else if (option.threshold<0.97) {
            option.wordLength = 6;
        } else {
            option.wordLength = 5;
        }
    } else {
        if (option.wordLength<4 || option.wordLength>12) {
            std::cout << "word length out of range" << std::endl;
            std::cout << "4<=word length<=12" << std::endl;
            printUsage();
            exit(0);
        }
    }
    if (option.drop == 3) {  
        if (option.threshold <= 0.879) {
            option.drop = 1;
        } else {
            option.drop = 0;
        }
    } else {
        if (option.drop != 0 && option.drop != 1) {
            std::cout << "drop error" << std::endl;
            std::cout << "drop=0/1" << std::endl;
            printUsage();
            exit(0);
        }
    }
    if (option.pigeon != 1 && option.pigeon != 0) {  
        std::cout << "pigeon error" << std::endl;
        std::cout << "pigeon=0/1" << std::endl;
        printUsage();
        exit(0);
    }
    std::cout << "input file:\t" << option.inputFile << std::endl;
    std::cout << "output file:\t" << option.outputFile << std::endl;
    std::cout << "threshold:\t" << option.threshold << std::endl;
    std::cout << "word length:\t" << option.wordLength << std::endl;
    std::cout << "drop filter\t" << option.drop << std::endl;
}

void readFile(std::vector<Read> &reads, Option &option) {
    std::ifstream file(option.inputFile);
    Read read;
    std::string line;
    getline(file, line);
    read.name = line;
    while (getline(file, line)) {  
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
    std::sort(reads.begin(), reads.end(), [](Read &a, Read &b) {
        return a.data.size() > b.data.size();  
    });
    std::cout << "longest:\t" << reads[0].data.size() << std::endl;
    std::cout << "shortest:\t" << reads[reads.size()-1].data.size() << std::endl;
    std::cout << "reads count：\t" << reads.size() << std::endl;
}

void copyData(std::vector<Read> &reads, Data &data) {
    int readsCount = reads.size();
    data.readsCount = readsCount;
    cudaMallocManaged(&data.lengths, readsCount*sizeof(int));
    cudaMallocManaged(&data.offsets, (readsCount+1)*sizeof(long));
    data.offsets[0] = 0;
    for (int i=0; i<readsCount; i++) {  
        int length = reads[i].data.size();
        data.lengths[i] = length;
        data.offsets[i+1] = data.offsets[i] + length/16*16+16;
    }
    cudaMallocManaged(&data.reads, data.offsets[readsCount]*sizeof(char));
    for (int i=0; i<readsCount; i++) {  
        int offset = data.offsets[i];
        int length = data.lengths[i];
        memcpy(&data.reads[offset], reads[i].data.c_str(), length*sizeof(char));
    }
    cudaDeviceSynchronize();  
}

void __global__ kernel_baseToNumber(char *reads, long length) {
    long index = threadIdx.x+blockDim.x*blockIdx.x;
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
    long length = data.offsets[readsCount];  
    kernel_baseToNumber<<<128, 128>>>(data.reads, length);
    cudaDeviceSynchronize();  
}

void __global__ kernel_createPigeon1(char *reads, int *lengths, long *offsets,
unsigned short *pigeon, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  
    int start = offsets[index];
    int end = start + lengths[index];
    int offset = index*256;
    for (int i=0; i<256; i++) {
        pigeon[offset+i] = 0;
    }
    char bases[4];  
    for(int i=0; i<4; i++) {  
        bases[i] = 4;
    }
    for (int i=start; i<end; i++) {
        for(int j=0; j<3; j++) {  
            bases[j] = bases[j+1];
        }
        bases[3] = reads[i];
        unsigned short indexValue = 0;
        int flag = 0;  
        for (int j=0; j<4; j++) {
            indexValue += (bases[j]&3)<<(3-j)*2;
            flag += max(bases[j]-3, 0);
        }
        if (!flag) {
            pigeon[offset+indexValue] += 1;
        }
    }
    for (int i=1; i<256; i++) {
        pigeon[offset+i] += pigeon[offset+i-1];
    }
}

void __global__ kernel_createPigeon2(char *reads, int *lengths, long *offsets,
unsigned short *pigeon, unsigned short *pigeonIndex, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  
    long start = offsets[index];
    long end = start + lengths[index];
    int offset = index*256;
    char bases[4];  
    for(int i=0; i<4; i++) {  
        bases[i] = 4;
    }
    for (int i=start; i<end; i++) {
        for(int j=0; j<3; j++) {  
            bases[j] = bases[j+1];
        }
        bases[3] = reads[i];
        unsigned short indexValue = 0;
        int flag = 0;  
        for (int j=0; j<4; j++) {
            indexValue += (bases[j]&3)<<(3-j)*2;
            flag += max(bases[j]-3, 0);
        }
        if (!flag) {  
            pigeon[offset+indexValue] -= 1;
            int temp = pigeon[offset+indexValue];
            pigeonIndex[start+temp] = i;
        }
    }
}

void createPigeon(Data &data) {
    int readsCount = data.readsCount;
    int length = data.offsets[readsCount];
    cudaMallocManaged(&data.pigeon,
        256*readsCount*sizeof(unsigned short));  
    memset(data.pigeon, 0, 256*readsCount*sizeof(unsigned short));  
    cudaMallocManaged(&data.pigeonIndex,
        length*sizeof(unsigned short));  
    memset(data.pigeonIndex, 0, length*sizeof(unsigned short));  
    kernel_createPigeon1<<<(readsCount+127)/128, 128>>>
    (data.reads, data.lengths, data.offsets, data.pigeon, readsCount);  
    kernel_createPigeon2<<<(readsCount+127)/128, 128>>>
    (data.reads, data.lengths, data.offsets, data.pigeon,
    data.pigeonIndex, readsCount);  
    kernel_createPigeon1<<<(readsCount+127)/128, 128>>>
    (data.reads, data.lengths, data.offsets, data.pigeon, readsCount);  
    cudaDeviceSynchronize();  
}


void __global__ kernel_compressData(int *lengths, long *offsets, char *reads,
unsigned int *compressed, int *gaps, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  
    long mark = offsets[index]/16;  
    int round = 0;  
    int gapCount = 0;  
    unsigned int compressedTemp = 0;  
    long start = offsets[index];
    long end = start + lengths[index];
    for (long i=start; i<end; i++) {
        unsigned char base = reads[i];  
        if (base < 4) {  
            compressedTemp += base << (15-round)*2;
            round++;
            if (round == 16) {
                compressed[mark] = compressedTemp;
                compressedTemp = 0;
                round = 0;
                mark++;
            }
        } else {  
            gapCount++;
        }
    }
    compressed[mark] = compressedTemp;
    gaps[index] = gapCount;
}

void compressData(Data &data) {
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];
    cudaMallocManaged(&data.compressed, length/16*sizeof(unsigned int));
    cudaMallocManaged(&data.gaps, readsCount*sizeof(int));
    kernel_compressData<<<(readsCount+127)/128, 128>>> (data.lengths,
    data.offsets, data.reads, data.compressed, data.gaps, readsCount);
    cudaFree(data.reads);
    cudaDeviceSynchronize();  
}

void __global__ kernel_createCutoff( int *lengths,
int *wordCutoff, int *baseCutoff, int *pigeonCutoff,
float threshold, int wordLength, int readsCount) {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= readsCount) return;  
    
    int length = lengths[index];
    int required = length - wordLength + 1;
    int cutoff = ceil((float)length*(1.0-threshold))*wordLength;
    required -= cutoff;
    required = max(required, 1);
    float offset = 0;  
    if (threshold >= 0.9) {
        offset = 1.1-fabs(threshold-0.95)*2;
    } else {
        offset = 1;
    }
    required = ceil((float)required*offset);
    wordCutoff[index] = required;
    
    required = ceil((float)length*threshold);
    baseCutoff[index] = required;
    
    cutoff = ceil((float)length*(1.0-threshold))*4;
    pigeonCutoff[index] = cutoff;
}

void createCutoff(Data &data, Option &option) {
    int readsCount = data.readsCount;
    cudaMallocManaged(&data.wordCutoff, readsCount*sizeof(int));  
    cudaMallocManaged(&data.baseCutoff, readsCount*sizeof(int));  
    cudaMallocManaged(&data.pigeonCutoff, readsCount*sizeof(int));  

    kernel_createCutoff<<<(readsCount+127)/128, 128>>>
    (data.lengths, data.wordCutoff, data.baseCutoff, data.pigeonCutoff,
    option.threshold, option.wordLength, readsCount);
    cudaDeviceSynchronize();  
}

void __global__ kernel_createPrefix(int *lengths, long *offsets,
unsigned int *compressed, int *gaps, int *prefix, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  
    int length = lengths[index] - gaps[index];  
    long start = offsets[index]/16;  
    int base[4];  
    for (int i=0; i<4; i++) {  
        base[i] = 0;
    }
    int count = 0;  
    for (long i=start;; i++) {  
        unsigned int zip = compressed[i];  
        unsigned int unzip = 0;  
        for (int j=0; j<16; j++) {  
            unzip = 3 & zip >> (15-j)*2;
            switch (unzip) {
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
            }
            count += 1;
            if (count == length) {
                break;
            }
        }
        if (count == length) {
            break;
        }
    }
    prefix[index*4+0] = base[0];
    prefix[index*4+1] = base[1];
    prefix[index*4+2] = base[2];
    prefix[index*4+3] = base[3];
}

void createPrefix(Data &data) {
    int readsCount = data.readsCount;
    cudaMallocManaged(&data.prefix, readsCount*4*sizeof(int));  
    kernel_createPrefix<<<(readsCount+127)/128, 128>>>
    (data.lengths, data.offsets, data.compressed,
    data.gaps, data.prefix, readsCount);  
    cudaDeviceSynchronize();
}

void __global__ kernel_createWords5(int *lengths, long *offsets,
unsigned int *compressed, int *gaps, unsigned int *words, int *wordCounts,
int readsCount) {
    int wordLength = 5;
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  
    long baseStart = offsets[index]/16;  
    long wordStart = offsets[index];  
    int length = lengths[index] - gaps[index];  
    int count = 0;  
    unsigned int base = 0;  
    unsigned int word = 0;  
    unsigned int mask = 0;  
    for (int i=0; i<wordLength; i++) {  
        mask += 3<<i*2;
    }
    for (long i=baseStart;; i++) {  
        for (int j=0; j<16; j++) {  
            base = compressed[i]>>(15-j)*2&3;
            word = word<<2;
            word = word + base;
            word = word & mask;
            words[wordStart+count] = word;
            count += 1;
            if (count == length) {
                break;
            }
        }
        if (count == length) {
            break;
        }
    }
    length = length - wordLength + 1;  
    wordCounts[index] = length;  
    for (int i=0; i<length; i++) {  
        words[wordStart+i] = words[wordStart+i+wordLength-1];
    }
}

void __global__ kernel_createWords6(int *lengths, long *offsets,
unsigned int *compressed, int *gaps, unsigned int *words, int *wordCounts,
int readsCount) {
    int wordLength = 6;
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  
    long baseStart = offsets[index]/16;  
    long wordStart = offsets[index];  
    int length = lengths[index] - gaps[index];  
    int count = 0;  
    unsigned int base = 0;  
    unsigned int word = 0;  
    unsigned int mask = 0;  
    for (int i=0; i<wordLength; i++) {  
        mask += 3<<i*2;
    }
    for (long i=baseStart;; i++) {  
        for (int j=0; j<16; j++) {  
            base = compressed[i]>>(15-j)*2&3;
            word = word<<2;
            word = word + base;
            word = word & mask;
            words[wordStart+count] = word;
            count += 1;
            if (count == length) {
                break;
            }
        }
        if (count == length) {
            break;
        }
    }
    length = length - wordLength + 1;  
    wordCounts[index] = length;  
    for (int i=0; i<length; i++) {  
        words[wordStart+i] = words[wordStart+i+wordLength-1];
    }
}

void __global__ kernel_createWords7(int *lengths, long *offsets,
unsigned int *compressed, int *gaps, unsigned int *words, int *wordCounts,
int readsCount) {
    int wordLength = 7;
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  
    long baseStart = offsets[index]/16;  
    long wordStart = offsets[index];  
    int length = lengths[index] - gaps[index];  
    int count = 0;  
    unsigned int base = 0;  
    unsigned int word = 0;  
    unsigned int mask = 0;  
    for (int i=0; i<wordLength; i++) {  
        mask += 3<<i*2;
    }
    for (long i=baseStart;; i++) {  
        for (int j=0; j<16; j++) {  
            base = compressed[i]>>(15-j)*2&3;
            word = word<<2;
            word = word + base;
            word = word & mask;
            words[wordStart+count] = word;
            count += 1;
            if (count == length) {
                break;
            }
        }
        if (count == length) {
            break;
        }
    }
    length = length - wordLength + 1;  
    wordCounts[index] = length;  
    for (int i=0; i<length; i++) {  
        words[wordStart+i] = words[wordStart+i+wordLength-1];
    }
}

void __global__ kernel_createWords(int *lengths, long *offsets,
unsigned int *compressed, int *gaps, unsigned int *words, int *wordCounts,
int readsCount, int wordLength) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  
    long baseStart = offsets[index]/16;  
    long wordStart = offsets[index];  
    int length = lengths[index] - gaps[index];  
    int count = 0;  
    unsigned int base = 0;  
    unsigned int word = 0;  
    unsigned int mask = 0;  
    for (int i=0; i<wordLength; i++) {  
        mask += 3<<i*2;
    }
    for (long i=baseStart;; i++) {  
        for (int j=0; j<16; j++) {  
            base = compressed[i]>>(15-j)*2&3;
            word = word<<2;
            word = word + base;
            word = word & mask;
            words[wordStart+count] = word;
            count += 1;
            if (count == length) {
                break;
            }
        }
        if (count == length) {
            break;
        }
    }
    length = length - wordLength + 1;  
    wordCounts[index] = length;  
    for (int i=0; i<length; i++) {  
        words[wordStart+i] = words[wordStart+i+wordLength-1];
    }
}

void createWords(Data &data, Option &option) {
    int readsCount = data.readsCount;
    int wordLength = option.wordLength;
    int length = data.offsets[readsCount];
    cudaMallocManaged(&data.words, length*sizeof(unsigned int));  
    cudaMallocManaged(&data.wordCounts, readsCount*sizeof(int));  
    switch (option.wordLength) {
        case 5:
            kernel_createWords5<<<(readsCount+127)/128, 128>>>
            (data.lengths, data.offsets, data.compressed, data.gaps,
            data.words, data.wordCounts, readsCount);  
            break;
        case 6:
            kernel_createWords6<<<(readsCount+127)/128, 128>>>
            (data.lengths, data.offsets, data.compressed, data.gaps,
            data.words, data.wordCounts, readsCount);  
            break;
        case 7:
            kernel_createWords7<<<(readsCount+127)/128, 128>>>
            (data.lengths, data.offsets, data.compressed, data.gaps,
            data.words, data.wordCounts, readsCount);  
            break;
        default:
            kernel_createWords<<<(readsCount+127)/128, 128>>>
            (data.lengths, data.offsets, data.compressed, data.gaps,
            data.words, data.wordCounts, readsCount, wordLength);  
            break;
    }
    cudaDeviceSynchronize();  
}

void __global__ kernel_sortWords(long *offsets, int *gaps,  unsigned int *words,
int *wordCounts, int wordLength, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
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
    kernel_sortWords<<<(readsCount+127)/128, 128>>>
    (data.offsets, data.gaps, data.words, data.wordCounts,
    option.wordLength, readsCount);  
    cudaDeviceSynchronize();  
}

void __global__ kernel_mergeWords(long *offsets, unsigned int *words,
int *wordCounts, unsigned short *orders, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
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
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];
    cudaMallocManaged(&data.orders, length*sizeof(unsigned short));  
    kernel_mergeWords<<<(readsCount+127)/128, 128>>>
    (data.offsets, data.words, data.wordCounts, data.orders, readsCount);
    cudaDeviceSynchronize();  
}

void initBench(Bench &bench, int readsCount) {
    cudaMallocManaged(&bench.table, (1<<2*12)*sizeof(unsigned short));  
    memset(bench.table, 0, (1<<2*12)*sizeof(unsigned short));  
    cudaMallocManaged(&bench.cluster, readsCount*sizeof(int));  
    for (int i=0; i<readsCount; i++) {  
        bench.cluster[i] = -1;
    }
    cudaMallocManaged(&bench.remainList, readsCount*sizeof(int));  
    for (int i=0; i<readsCount; i++) {  
        bench.remainList[i] = i;
    }
    bench.remainCount = readsCount;  
    cudaMallocManaged(&bench.jobList, readsCount*sizeof(int));  
    for (int i=0; i<readsCount; i++) {  
        bench.jobList[i] = i;
    }
    bench.jobCount = readsCount;  
    bench.representative = -1;
    cudaDeviceSynchronize();  
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

void updataRemain(int *cluster, int *remainList, int &remainCount) {
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
        if (jobList[i] >= 0) {
            jobList[count] = jobList[i];
            count += 1;
        }
    }
    jobCount = count;
}

void __global__ kernel_makeTable(long *offsets, unsigned int *words,
int *wordCounts, unsigned short *orders,
unsigned short *table, int representative) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    long start = offsets[representative];
    int length = wordCounts[representative];
    for (int i=index; i<length; i+=128*128) {
        unsigned int word = words[start+i];
        unsigned short order = orders[start+i];
        if (order > 0) table[word] = order;  
    }
}

void __global__ kernel_cleanTable(long *offsets, unsigned int *words,
int* wordCounts, unsigned short *orders,
unsigned short *table, int representative) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    long start = offsets[representative];
    int length = wordCounts[representative];
    for (int i=index; i<length; i+=128*128) {
        unsigned int word = words[start+i];
        unsigned short order = orders[start+i];
        if (order > 0) table[word] = 0;  
    }
}

void __global__ kernel_preFilter(int *prefix, int *baseCutoff,
int *jobList, int jobCount, int representative) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= jobCount) return;  
    int query = jobList[index];  
    int offsetOne = representative*4;  
    int offsetTwo = query*4;  
    int sum = 0;
    sum += min(prefix[offsetOne+0], prefix[offsetTwo+0]);
    sum += min(prefix[offsetOne+1], prefix[offsetTwo+1]);
    sum += min(prefix[offsetOne+2], prefix[offsetTwo+2]);
    sum += min(prefix[offsetOne+3], prefix[offsetTwo+3]);
    int cutoff = baseCutoff[query];
    if (sum < cutoff) {  
        jobList[index] = -1;
    }
}

void __global__ kernel_pigeon(float threshold, int *lengths, long *offsets,
int *pigeonCutoff, unsigned short *pigeonIndex, unsigned short *pigeon,
int *jobList, int jobCount, int representative) {
    int index = jobList[blockIdx.x];
    __shared__ int sum[256];  
    sum[threadIdx.x] = 0;
    
    long offsetOne = representative * 256;
    long offsetTwo = index * 256;
    int startOne = 0;
    int startTwo = 0;
    if (threadIdx.x > 0) {  
        startOne = pigeon[offsetOne+threadIdx.x-1];
        startTwo = pigeon[offsetTwo+threadIdx.x-1];
    }
    int endOne = pigeon[offsetOne+threadIdx.x];
    int endTwo = pigeon[offsetTwo+threadIdx.x];
    offsetOne = offsets[representative];
    offsetTwo = offsets[index];
    int lengthOne = lengths[representative];  
    int lengthTwo = lengths[index];  
    int gap = lengthOne - ceil(lengthTwo*threshold);  
    while (startOne < endOne && startTwo < endTwo) {  
        int one = pigeonIndex[offsetOne+startOne];
        int two = pigeonIndex[offsetTwo+startTwo];
        if (abs(one-two) >= gap) {
            startOne += 1;
            startTwo += 1;
            sum[threadIdx.x] += 1;
        } else if (one > two) {
            startOne += 1;
        } else {
            startTwo += 1;
        }
    }
    __syncthreads();  
    for (int i=256/2; i>0; i/=2) {  
        if (threadIdx.x>=i) return;  
        sum[threadIdx.x] += sum[threadIdx.x+i];
        __syncthreads();
    }
    int cutoff = pigeonCutoff[index];
    if (sum[0] < cutoff) {  
        jobList[blockIdx.x] = -1;
    }
}

void __global__ kernel_filter(long *offsets, unsigned int *words,
int *wordCounts, unsigned short *orders, int *wordCutoff,
unsigned short *table, int *jobList, int jobCount) {
    if (blockIdx.x >= jobCount) return;  
    int index = jobList[blockIdx.x];  
    __shared__ int result[128];  
    result[threadIdx.x] = 0;  
    long start = offsets[index];
    int length = wordCounts[index];
    for (int i=threadIdx.x; i<length; i+=128) {
        unsigned int value = words[start+i];  
        result[threadIdx.x] += min(table[value], orders[start+i]);
    }
    __syncthreads();  
    for (int i=128/2; i>0; i/=2) {  
        if (threadIdx.x>=i) return;  
        result[threadIdx.x] += result[threadIdx.x+i];
        __syncthreads();
    }
    int cutoff = wordCutoff[index];
    if(result[0] < cutoff) {  
        jobList[blockIdx.x] = -1;
    }
}

__global__ void kernel_align(float threshold, int *lengths, long *offsets,
unsigned int *compressed, int *gaps, int *baseCutoff, int *cluster,
int *jobList, int jobCount, int representative) {
    
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= jobCount) return;  
    index = jobList[index];
    int target = representative;  
    int query = index;  
    int cutoff = baseCutoff[index];
    int targetLength = lengths[target] - gaps[target];  
    int queryLength = lengths[query] - gaps[query];  
    int target32Length = targetLength/16+1;  
    int query32Length  = queryLength/16+1;  
    int targetOffset = offsets[target]/16;  
    int queryOffset = offsets[query]/16;  
    short rowNow[MAX_LENGTH] = {0};  
    short rowPrevious[MAX_LENGTH] = {0};  
    int columnPrevious[17] = {0};  
    int columnNow[17] = {0};  
    int shift = ceil((float)targetLength-(float)queryLength*threshold);
    shift = ceil((float)shift/16.0);  
    
    for (int i = 0; i < query32Length; i++) {  
        
        for (int j=0; j<17; j++) {  
            columnPrevious[j] = 0;
            columnNow[j] = 0;
        }
        int targetIndex = 0;  
        unsigned int queryPack = compressed[queryOffset+i];  
        int jstart = i-shift;
        jstart = max(jstart, 0);
        int jend = i+shift;
        jend = min(jend, target32Length);
        for (int j=0; j<target32Length; j++) {  
            columnPrevious[0] = rowPrevious[targetIndex];
            unsigned int targetPack = compressed[targetOffset+j];  
            
            for (int k=30; k>=0; k-=2) {  
                
                int targetBase = (targetPack>>k)&3;  
                int m=0;
                columnNow[m] = rowPrevious[targetIndex+1];
                for (int l=30; l>=0; l-=2) {  
                    m++;
                    int queryBase = (queryPack>>l)&3;  
                    int diffScore = queryBase == targetBase;
                    columnNow[m] = columnPrevious[m-1] + diffScore;
                    columnNow[m] = max(columnNow[m], columnNow[m-1]);
                    columnNow[m] = max(columnNow[m], columnPrevious[m]);
                }
                targetIndex++;
                rowNow[targetIndex] = columnNow[16];
                if (targetIndex == targetLength) {  
                    if(i == query32Length-1) {  
                        int score = columnNow[queryLength%16];
                        if (score >= cutoff) {
                            cluster[index] = target;
                        } else {
                            index = threadIdx.x + blockDim.x*blockIdx.x;
                            jobList[index] = -1;
                        }
                        return;
                    }
                    break;
                }
                
                k-=2;
                targetBase = (targetPack>>k)&3;  
                m=0;
                columnPrevious[m] = rowPrevious[targetIndex+1];
                for (int l=30; l>=0; l-=2) {  
                    m++;
                    int queryBase = (queryPack>>l)&3;  
                    int diffScore = queryBase == targetBase;
                    columnPrevious[m] = columnNow[m-1] + diffScore;
                    columnPrevious[m] = max(columnPrevious[m], columnPrevious[m-1]);
                    columnPrevious[m] = max(columnPrevious[m], columnNow[m]);
                }
                targetIndex++;
                rowNow[targetIndex] = columnPrevious[16];
                if (targetIndex == targetLength) {  
                    if(i == query32Length-1) {  
                        int score = columnPrevious[queryLength%16];
                        if (score >= cutoff) {
                            cluster[index] = target;
                        } else {
                            index = threadIdx.x + blockDim.x*blockIdx.x;
                            jobList[index] = -1;
                        }
                        return;
                    }
                    break;
                }
            }
        }
        
        i++;
        for (int j=0; j<17; j++) {  
            columnPrevious[j] = 0;
            columnNow[j] = 0;
        }
        targetIndex = 0;  
        queryPack = compressed[queryOffset+i];  
        jstart = i-shift;
        jstart = max(jstart, 0);
        jend = i+shift;
        jend = min(jend, target32Length);
        for (int j=0; j<target32Length; j++) {  
            unsigned int targetPack = compressed[targetOffset+j];  
            
            for (int k=30; k>=0; k-=2) {  
                
                int targetBase = (targetPack>>k)&3;  
                int m=0;
                columnNow[m] = rowNow[targetIndex+1];
                for (int l=30; l>=0; l-=2) {  
                    m++;
                    int queryBase = (queryPack>>l)&3;  
                    int diffScore = queryBase == targetBase;
                    columnNow[m] = columnPrevious[m-1] + diffScore;
                    columnNow[m] = max(columnNow[m], columnNow[m-1]);
                    columnNow[m] = max(columnNow[m], columnPrevious[m]);
                }
                targetIndex++;
                rowPrevious[targetIndex] = columnNow[16];
                if (targetIndex == targetLength) {  
                    if(i == query32Length-1) {  
                        int score = columnNow[queryLength%16];
                        if (score >= cutoff) {
                            cluster[index] = target;
                        } else {
                            index = threadIdx.x + blockDim.x*blockIdx.x;
                            jobList[index] = -1;
                        }
                        return;
                    }
                    break;
                }
                
                k-=2;
                targetBase = (targetPack>>k)&3;  
                m=0;
                columnPrevious[m] = rowNow[targetIndex+1];
                for (int l=30; l>=0; l-=2) {  
                    m++;
                    int queryBase = (queryPack>>l)&3;  
                    int diffScore = queryBase == targetBase;
                    columnPrevious[m] = columnNow[m-1] + diffScore;
                    columnPrevious[m] = max(columnPrevious[m], columnPrevious[m-1]);
                    columnPrevious[m] = max(columnPrevious[m], columnNow[m]);
                }
                targetIndex++;
                rowPrevious[targetIndex] = columnPrevious[16];
                if (targetIndex == targetLength) {  
                    if(i == query32Length-1) {  
                        int score = columnPrevious[queryLength%16];
                        if (score >= cutoff) {
                            cluster[index] = target;
                        } else {
                            index = threadIdx.x + blockDim.x*blockIdx.x;
                            jobList[index] = -1;
                        }
                        return;
                    }
                    break;
                }
            }
        }
    }
}

void clustering(Option &option, Data &data, Bench &bench) {
    int readsCount = data.readsCount;
    initBench(bench, readsCount);  
    std::cout << "now/whole:\t" << std::endl;
    while (true) {  
        
        
        updateRepresentative(bench.cluster, bench.representative, readsCount);
        if (bench.representative >= readsCount) break;  
        
        std::cout << "\r" << bench.representative+1 << "/" << readsCount;
        std::flush(std::cout);  
        
        updataRemain(bench.cluster, bench.remainList, bench.remainCount);
        
        cudaDeviceSynchronize();  
        memcpy(bench.jobList, bench.remainList, bench.remainCount*sizeof(int));
        bench.jobCount = bench.remainCount;
        cudaDeviceSynchronize();  
        
        kernel_makeTable<<<128, 128>>>(data.offsets, data.words,
        data.wordCounts, data.orders, bench.table, bench.representative);
        cudaDeviceSynchronize();  
        
        if (bench.jobCount > 0) {  
            kernel_preFilter<<<(bench.jobCount+127)/128, 128>>>
            (data.prefix, data.baseCutoff, bench.jobList,
            bench.jobCount, bench.representative);
        }
        cudaDeviceSynchronize();  
        if (option.pigeon == 1) {
            updatJobs(bench.jobList, bench.jobCount);  
            if (bench.jobCount > 0) {  
                kernel_pigeon<<<bench.jobCount, 256>>>
                (option.threshold, data.lengths, data.offsets,
                data.pigeonCutoff, data.pigeonIndex, data.pigeon,
                bench.jobList, bench.jobCount, bench.representative);
            }
            cudaDeviceSynchronize();  
        }
        updatJobs(bench.jobList, bench.jobCount);  
        if (bench.jobCount > 0) {  
            kernel_filter<<<bench.jobCount, 128>>>
            (data.offsets, data.words, data.wordCounts, data.orders,
            data.wordCutoff, bench.table, bench.jobList, bench.jobCount);
        }
        cudaDeviceSynchronize();  

        
        updatJobs(bench.jobList, bench.jobCount);  
        if (bench.jobCount > 0) {  
            kernel_align<<<(bench.jobCount+127)/128, 128>>>(option.threshold,
            data.lengths, data.offsets, data.compressed,
            data.gaps, data.baseCutoff, bench.cluster,
            bench.jobList, bench.jobCount, bench.representative);
        }
        cudaDeviceSynchronize();  
        updatJobs(bench.jobList, bench.jobCount);  
        
        
        kernel_cleanTable<<<128, 128>>>(data.offsets, data.words,
        data.wordCounts, data.orders, bench.table, bench.representative);
        cudaDeviceSynchronize();  
    }
    std::cout << std::endl;
}

void clusteringDrop(Option &option, Data &data, Bench &bench) {
    int readsCount = data.readsCount;
    initBench(bench, readsCount);  
    std::cout << "now/whole:\t" << std::endl;
    while (true) {  
        
        
        updateRepresentative(bench.cluster, bench.representative, readsCount);
        if (bench.representative >= readsCount) break;  
        
        std::cout << "\r" << bench.representative+1 << "/" << readsCount;
        std::flush(std::cout);  
        
        updataRemain(bench.cluster, bench.remainList, bench.remainCount);
        
        memcpy(bench.jobList, bench.remainList, bench.remainCount*sizeof(int));
        bench.jobCount = bench.remainCount;

        
        updatJobs(bench.jobList, bench.jobCount);  
        if (bench.jobCount > 0) {  
            kernel_align<<<(bench.jobCount+127)/128, 128>>>(option.threshold,
            data.lengths, data.offsets, data.compressed,
            data.gaps, data.baseCutoff, bench.cluster,
            bench.jobList, bench.jobCount, bench.representative);
        }
        cudaDeviceSynchronize();  
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
    std::cout << "cluster count:\t" << sum << std::endl;
}

void checkValue(Option &option, Data &data, Bench &bench) {
    cudaDeviceSynchronize();  
    cudaDeviceSynchronize();  
}

void checkError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }
}
