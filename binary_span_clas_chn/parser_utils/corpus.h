//
// Created by ji_ma on 4/9/17.
//

#ifndef NEURAL_CKY_CORPUS_H
#define NEURAL_CKY_CORPUS_H
#pragma once

#include "dynet/dict.h"

#include <unordered_map>
#include <unordered_set>
#include <fstream>

#include "cl-args.h"
#include "id_builder.h"

using namespace dynet;
using namespace std;


void ReadCorpus(const string& gold_file,  vector<ConTree*>& gold_trees)
{
    cerr << "Reading data from " << gold_file << "...\n";

    ifstream gold_in(gold_file);

    assert(gold_in);

    string gold_line;

    while(getline(gold_in, gold_line)) {

        ConTree* gold_tree = new ConTree(gold_line);

        /*if(gold_tree->size() < 6)
        {
            cout << gold_line << endl;
            cout << (*gold_tree) << endl;
        }*/

        gold_trees.push_back(gold_tree);
        //exit(0);
    }
    gold_in.close();

    cerr<< "[log]: #Trees: " << gold_trees.size() << endl;
}

void ReadUnkMap(const string& input_file, IdBuilder* id_builder)
{
    unordered_map<string, string>& unk_map = id_builder->get_unk_map();
    ifstream fin(input_file);
    assert(fin);
    string word, unkword;
    while(fin>>word>>unkword) {
        unk_map[word] = unkword;
        if(id_builder->can_add_unk_types_to_dict()) {
            id_builder->get_all_dict().token_dict.dict.convert(unkword);
        }
    }
    fin.close();
}


void ReadEmbeddings(struct Params& params, IdBuilder* id_builder)
{
    cerr << "Reading pretrained vector from " << params.embedding_file<< "...\n";
    ifstream fin(params.embedding_file);
    assert(fin);
    string line;

    Dict& word_dict = id_builder->get_all_dict().token_dict.dict;

    unordered_map<string, vector<float> >& pretrained_embeddings = id_builder->get_pretrained_embeddings();

    vector<float>& averaged_vec = id_builder->get_averaged_vec();

    bool first = true;
    while(getline(fin, line))
    {
        istringstream sin(line);
        string word;
        sin>>word;
        float value;
        vector<float> vecs;
        while(sin>>value)
            vecs.push_back(value);
        if(first)
        {
            first = false;
            params.TOKEN_INPUT_DIM = vecs.size();
            averaged_vec.resize(params.TOKEN_INPUT_DIM, 0.0);
        }
        for(unsigned i = 0u; i< params.TOKEN_INPUT_DIM; i++)
        {
            averaged_vec[i] += vecs[i];
        }
        pretrained_embeddings[word] = vecs;
        if(id_builder->can_add_pretrained_types_to_dict())
            word_dict.convert(word);
    }
    fin.close();
    for(unsigned i = 0u; i< params.TOKEN_INPUT_DIM; i++)
    {
        averaged_vec[i] /= pretrained_embeddings.size();
    }
}

#endif //NEURAL_CKY_CORPUS_H
