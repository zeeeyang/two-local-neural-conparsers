//
// Created by zeeeyang on 4/9/17.
//

#ifndef NEURAL_CKY_TREENODE_H
#define NEURAL_CKY_TREENODE_H
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>

#include "dynet/dynet.h"
#include "dynet/expr.h"

using namespace std;
using namespace dynet::expr;
using namespace dynet;

struct InputNode {

    unsigned word_id;
    unsigned original_word_id;
    unsigned unk_id;

    string word;
    string lowercased_word;

    vector<string>  chars;
    vector<unsigned>  char_ids;

    string postag;
    unsigned postag_id;


    bool is_single;


    InputNode(): is_single(false)
    {

    }
};


struct PhraseNode {

    unsigned span_label_id;
    string span_label;
    string direction;
    string simple_span_label;
    Expression span_label_xi;

    PhraseNode()
    {

    }
};

struct SpanTreeNode {

    vector<struct SpanTreeNode*> children;
    vector<struct PhraseNode*> phrase_nodes;

    Expression xi;
    bool is_leaf;

    int span_left;
    int span_right;

    Expression bu_hi;
    Expression bu_ci;


    SpanTreeNode(): is_leaf(false), span_left(-1), span_right(-1)
    {
        children.clear();
    }
    ~SpanTreeNode() {
        if(children.size() == 0) return;
        for(auto& child: children)
        {
            if(child !=NULL)
                delete child;
            child = NULL;
        }
        for(auto& p: phrase_nodes)
        {
            if(p !=NULL)
                delete p;
            p = NULL;
        }
    }
};
#endif //NEURAL_CKY_TREENODE_H
