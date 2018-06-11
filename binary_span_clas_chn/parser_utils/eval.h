//
// Created by zeeeyang on 4/8/17.
//

#ifndef NEURAL_CKY_EVAL_H
#define NEURAL_CKY_EVAL_H
#pragma once

float exec(const char* cmd) {
    char buffer[1024];
    float score = 0.0;
    std::string result = "";
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer, 128, pipe.get()) != NULL)
            result += buffer;
    }
    istringstream sin(result);
    sin>>score;
    return score;
}

struct FMeasure {
    int num_total;
    int num_prediction;
    int num_common;
    int num_examples;
    string name;
    FMeasure(const string& _name): num_total(0), num_prediction(0), num_common(0), num_examples(0), name(_name)
    {

    }

    inline void reset() {
        num_total = 0;
        num_prediction = 0;
        num_common = 0;
        num_examples = 0;
    }

    inline float precision() const {
        return num_prediction == 0? 0 : (float) num_common / num_prediction;
    }

    inline float recall() const {
        return num_total == 0? 0: (float) num_common / num_total;
    }

    inline float f() const {
        int sum = num_total + num_prediction;
        return  sum == 0 ? 0:  2.0 * num_common / sum;
    }

    void report() const {
        cerr<< "[log] " << name << " FMeasure: " << endl;
        cerr << "[log]\t #Examples: " << num_examples << endl;
        cerr << "[log]\t #Counts: " << num_common << "\t" << num_prediction << "\t" << num_total << endl;
        cerr << "[log]\t #F1: " <<  precision() << "\t" << recall() << "\t" << f() << endl;
    }
};

struct AccMeasure {
    int num_total;
    int num_correct;
    AccMeasure(): num_total(0), num_correct(0)
    {

    }

    inline void reset() {
        num_total = 0;
        num_correct = 0;
    }

    inline float acc() const {
        return num_total == 0? 0: (float) num_correct/ num_total;
    }

    void report() const {
        cerr<< "[log] Accuracy: " << endl;
        cerr << "[log]\t #Counts: " << num_correct << "\t" << num_total << endl;
        cerr << "[log]\t #Acc: " <<  acc() << endl;
    }
};
#endif //NEURAL_CKY_EVAL_H
