#include <iostream>
#include "StopWatch.h"
#include <stdio.h>
#include <string.h>
#include "PatternMatch.h"
#include <fstream>
#include <omp.h>
using namespace std;

int badchar[256];
int globalcount;

char* loadFile(const char* fileName, int& fsize);
void ParallelMatchNaive(const char* txt, int N, const char* pat, int M);
void ParallelMatchBoyerMoore(const char* txt, int N, const char* pat, int M);



int main() {
    int MAXCHAR = 0;
    const char* text = loadFile("textdata.txt", MAXCHAR);

    if (text == nullptr) {
        printf("Failure at opening text file!\n");
        return 0;
    }

    int N = strlen(text);
    const int PATTERN_COUNT = 5;
    const char* pattern[PATTERN_COUNT] = { "and","fanny","removing", "possible humoured", "six hearted hundred towards" };

    int countNaive = 0;
    int countBoyer = 0;
    double eTimeNaive = 0;
    double eTimeBoyer = 0;
    StopWatch sw;

    ///////////////////////////////////////////////////////////////////////////
    // SERIAL PART
    ///////////////////////////////////////////////////////////////////////////
    printf("+------------------------------+---------------------+---------------------+\n");
    printf("|%-30s|%-21s|%-21s|\n", "SERIAL", "NAIVE", "BOYER_MOORE");
    printf("+------------------------------+---------------------+---------------------+\n");
    printf("|%-30s|%-10s|%-10s|%-10s|%-10s|\n", "Pattern", "Count", "Time(ms)", "Count", "Time(ms)");
    printf("+------------------------------+---------------------+---------------------+\n");
    for (int i = 0; i < PATTERN_COUNT; i++)
    {
        int M = strlen(pattern[i]);
        preprocessHeuristicArray(pattern[i], M, badchar);

        sw.start();
        countNaive = NaiveMatch(text, N, pattern[i], M);
        sw.stop();
        eTimeNaive = sw.elapsedTime();

        sw.start();
        countBoyer = BoyerMooreMatch(text, N, pattern[i], M, badchar);
        sw.stop();
        eTimeBoyer = sw.elapsedTime();

        printf("|%-30s|%-10d|%-10.4lf|%-10d|%-10.4lf|\n", pattern[i], countNaive, eTimeNaive, countBoyer, eTimeBoyer);
    }

    printf("+------------------------------+---------------------+---------------------+\n");
    printf("\n\n");


    ///////////////////////////////////////////////////////////////////////////
    // PARALLEL PART
    ///////////////////////////////////////////////////////////////////////////
    printf("+------------------------------+---------------------+---------------------+\n");
    printf("|%-30s|%-21s|%-21s|\n", "PARALLEL", "NAIVE", "BOYER_MOORE");
    printf("+------------------------------+---------------------+---------------------+\n");
    printf("|%-30s|%-10s|%-10s|%-10s|%-10s|\n", "Pattern", "Count", "Time(ms)", "Count", "Time(ms)");
    printf("+------------------------------+---------------------+---------------------+\n");
    for (int i = 0; i < PATTERN_COUNT; i++)
    {
        int M = strlen(pattern[i]);
        preprocessHeuristicArray(pattern[i], M, badchar);

        globalcount = 0;
        sw.start();
#pragma omp parallel num_threads(4)
        ParallelMatchNaive(text, N, pattern[i], M);
        sw.stop();
        eTimeNaive = sw.elapsedTime();
        countNaive = globalcount;


        globalcount = 0;
        sw.start();
#pragma omp parallel num_threads(4)
        ParallelMatchBoyerMoore(text, N, pattern[i], M);
        sw.stop();
        eTimeBoyer = sw.elapsedTime();
        countBoyer = globalcount;

        printf("|%-30s|%-10d|%-10.4lf|%-10d|%-10.4lf|\n", pattern[i], countNaive, eTimeNaive, countBoyer, eTimeBoyer);
    }
    printf("+------------------------------+---------------------+---------------------+\n");
    printf("\n\n");

}



void ParallelMatchNaive(const char* txt, int N, const char* pat, int M) {
    int thread_id = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    int fountCount = 0;

    int size = N / thread_count;
    int eIndex = (thread_id * size) + (size - 1);
    int sIndex = 0;

    if (thread_id != 0) {
        sIndex = (thread_id * size) - (M - 1);
        N = size + (M - 1);
    }
    else {
        sIndex = thread_id * size;
        if (thread_id != thread_count - 1) {
            N = size;
        }
        if (thread_id == thread_count - 1) {
            eIndex = N;
        }
    }

    for (int i = sIndex; i <= eIndex - M + 1; i++) {
        int j = 0;
        while ((j < M) && (txt[sIndex + j] == pat[j])) {
            j = j + 1;
        }
        if (j == M) {
            fountCount = fountCount + 1;
        }
        sIndex += 1;
    }

#  pragma omp critical
    globalcount += fountCount;
}

void ParallelMatchBoyerMoore(const char* txt, int N, const char* pat, int M) {
    int thread_id = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    int size = N / thread_count;
    int  sIndex = thread_id * size;
    int eIndex = (thread_id * size) + (size - 1);
    int localcount = 0;

    if (thread_id == thread_count - 1) {
        eIndex = N;
    }
    if (thread_id != 0) {
        sIndex = (thread_id * size) - (M - 1);
    }
    else {
        sIndex = thread_id * size;
        if (thread_id == thread_count - 1) {
            eIndex = N;
        }
    }
    localcount = BoyerMooreMatchWithRange(txt, sIndex, eIndex, pat, M, badchar);

# pragma omp critical
    globalcount += localcount;
}

char* loadFile(const char* fileName, int& fsize) {
    ifstream file;
    file.open(fileName);
    if (!file.is_open()) {
        cout << "File [" << fileName << "] could not open!" << endl;
        return nullptr;
    }
    const auto begin = file.tellg();
    file.seekg(0, ios::end);
    const auto end = file.tellg();
    fsize = (end - begin);
    char* content = new char[fsize + 1];
    file.seekg(0, ios::beg);
    file.read(content, fsize);
    file.close();
    content[fsize] = '\0';
    return content;
}
