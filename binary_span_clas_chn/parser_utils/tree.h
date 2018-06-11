//
// Created by ji_ma on 4/9/17.
//

#ifndef NEURAL_CKY_TREE_H
#define NEURAL_CKY_TREE_H
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

#include "treenode.h"
#include "all_dict.h"

//#include "example.h"

using namespace std;

typedef int SPAN_KEY;

class ConTree {
public:
    struct SpanTreeNode* root;
    vector<struct InputNode* > leaf_input_nodes;
    vector<struct SpanTreeNode* > leaf_spantree_nodes;
    unordered_map<SPAN_KEY, struct SpanTreeNode* > phrase_spantree_nodes;

    ConTree(const string& treerep) {

        leaf_input_nodes.clear();
        leaf_spantree_nodes.clear();

        root = build_tree(treerep);

        this->set_spans();
    }

    ConTree(int size, const unordered_map<SPAN_KEY, int>& chart_span_backtrace_pointers)
    {
        leaf_input_nodes.clear();
        leaf_spantree_nodes.clear();

        root = build_tree(0, size-1, size, chart_span_backtrace_pointers);
        // no set spanes
    }

    ~ConTree();


    void str(DictSet& all_dict, vector<string>& strs) {
        str(*root, all_dict, strs);
    }

    friend ostream& operator<<(ostream& cout, const ConTree& other)
    {
        auto slen = other.leaf_input_nodes.size();
        for(unsigned token_index = 0u; token_index < slen; token_index++) {
            cout<< other.leaf_input_nodes[token_index]->word  << "/" << other.leaf_input_nodes[token_index]->postag << " ";
        }
        if(slen)
            cout<<endl;
        for(auto& item: other.phrase_spantree_nodes)
        {
            int span_left = item.first / other.size();
            int span_right = item.first % other.size();
            cout<<"\t["<<span_left<<", "<<span_right<<"] ";
            for(auto span: item.second->phrase_nodes)
                cout<< "[ " << span->span_label <<  " ] ";
            cout<<endl;
        }
        return cout;
    }
    inline unsigned size() const {
        //for gold trees, size() == leaf_input_nodes.size() == leaf_spantree_nodes.size()
        //for predicted_trees, size() == leaf_spantree_nodes.size(), leaf_spantree_nodes.size() == 0
        return leaf_spantree_nodes.size();
    }
    void set_spans();
private:
    struct SpanTreeNode* build_tree(const string& treerep);
    struct SpanTreeNode* build_tree(int start, int end, int size, const unordered_map<SPAN_KEY, int>& chart_span_backtrace_pointers);
    void set_spans(struct SpanTreeNode& root);
    void str(struct SpanTreeNode& root, DictSet& all_dict, vector<string>& strs);
};

struct SpanTreeNode* ConTree::build_tree(const string& treerep)
{
    //cerr<< "[log1]: " << treerep << "\nlog2: " << treerep[0] << "\nlog3:" << treerep[treerep.size()-1] << endl;

    int i = 0;
    string remaining_rep = treerep;
    while(!remaining_rep.empty() && remaining_rep[i] == ' ')
        remaining_rep.erase(remaining_rep.begin());
    while(!remaining_rep.empty() && remaining_rep.back() == ' ')
        remaining_rep.erase(remaining_rep.begin()+remaining_rep.size()-1);
    //cerr<< "[log]: remaining1: " << remaining_rep << endl;
    if(remaining_rep[0] != '(' || remaining_rep[remaining_rep.size()-1] != ')') {
        cerr<<"[Error] mismatched brackets" << endl;
        return NULL;
    }
    remaining_rep = remaining_rep.substr(1, remaining_rep.size()-2);
    //cerr<< "[log]: remaining2: " << remaining_rep << endl;
    while(!remaining_rep.empty() && remaining_rep[i] == ' ')
        remaining_rep.erase(remaining_rep.begin());
    while(!remaining_rep.empty() && remaining_rep.back() == ' ')
        remaining_rep.erase(remaining_rep.begin()+remaining_rep.size()-1);
    //cerr<< "[log]: remaining3: " << remaining_rep << endl;

    size_t phraselabel_pos = remaining_rep.find(" ");
    const string& phraselabel = remaining_rep.substr(0, phraselabel_pos);
    const string& next_phrase_rep = remaining_rep.substr(phraselabel_pos+1);
    //cerr<< "[log]: next_phrase_rep: " << next_phrase_rep << endl;

    size_t dirlabel_pos = next_phrase_rep.find(" ");
    const string& dirlabel = next_phrase_rep.substr(0, dirlabel_pos);
    const string& next_rep = next_phrase_rep.substr(dirlabel_pos+1);

    //cerr<< "[log]: pr: " << phraselabel << " "<< endl;
    //cerr<< "[log]: next_rep: " << next_rep << endl;

    if( next_rep[0] != '(' ) // leaf node
    {
        struct InputNode* input_node = new struct InputNode;
        input_node->word = next_rep;
        //root->direction = dirlabel;
        if(next_rep == "-LRB-")
            input_node->word = "(";
        else if(next_rep == "-RRB-")
            input_node->word = ")";


        for(int i = 0; i < (int)input_node->word.size(); )
        {
            int charlen = UTF8Len(input_node->word[i]);
            input_node->chars.push_back(input_node->word.substr(i, charlen));
            i += charlen;
        }

        input_node->lowercased_word = input_node->word;
        input_node->postag = phraselabel;

        leaf_input_nodes.push_back(input_node);

        struct SpanTreeNode* root = new SpanTreeNode;
        root->is_leaf  = true;
        root->span_left = leaf_spantree_nodes.size();
        root->span_right = leaf_spantree_nodes.size();

        leaf_spantree_nodes.push_back(root);

        //cerr<<"[" << root.span_left << ", " << root.span_right << "] " << root.pos <<" " << root.word << endl;
        return root;
    }

    struct PhraseNode* phrase_node = new struct PhraseNode;
    phrase_node->simple_span_label = phraselabel;
    phrase_node->direction = dirlabel;
    if(phrase_node->direction.back() == '*')
        phrase_node->span_label = phraselabel + "*";
    else
        phrase_node->span_label = phraselabel;



    vector<string> children_reps;

    int left_b = 0, right_b = 0, start_pos = 0, end_pos = 0;
    int span_left = INT_MAX, span_right = INT_MIN;
    for(int i = 0; i< (int)next_rep.size(); i++)
    {
        if(next_rep[i] == '(')
        {
            left_b += 1;
            if(left_b == 1)
                start_pos = i;
        }
        else if( next_rep[i] == ')' )
        {
            right_b += 1;
            if( right_b == left_b )
            {
                end_pos = i;
                const string& child_rep = next_rep.substr(start_pos, end_pos-start_pos+1);
                children_reps.push_back(child_rep);

                left_b = 0;
                right_b = 0;
                start_pos = 0;
                end_pos = 0;
            }
        }
        else if(right_b > left_b)
        {
            cerr<<"[Error] mismatched brackets" << endl;
            return NULL;
        }
    }


    if(children_reps.size() == 2) {
        struct SpanTreeNode* root = new struct SpanTreeNode;
        for(auto& child_rep: children_reps) {
            //recursively call function
            struct SpanTreeNode* child_node = build_tree(child_rep);
            root->children.push_back(child_node);
            span_left = std::min(span_left, child_node->span_left);
            span_right = std::max(span_right, child_node->span_right);
        }
        root->span_left = span_left;
        root->span_right = span_right;
        root->phrase_nodes.push_back(phrase_node);
        return root;
    } else {
        assert(children_reps.size() == 1);
        auto& child_rep = children_reps[0];
        struct SpanTreeNode* child_node = build_tree(child_rep);

        child_node->phrase_nodes.push_back(phrase_node);

        assert(child_node->span_left != -1 && child_node->span_right != -1);
        return child_node;
    }
}

struct SpanTreeNode* ConTree::build_tree(int start, int end, int size, const unordered_map<SPAN_KEY, int>& chart_span_backtrace_pointers)
{
    if( start > end )
        return NULL;

    struct SpanTreeNode* root = new struct SpanTreeNode;
    root->span_left = start;
    root->span_right = end;

    if( start == end) {
        root->is_leaf = true;
        //cerr<<"[tlog1] " << start << ", " << end << endl;
        this->leaf_spantree_nodes.push_back(root);
        return root;
    }

    SPAN_KEY  span_key = start * size + end;
    int split_point =  chart_span_backtrace_pointers.find(span_key)->second;

    struct  SpanTreeNode* left_child = build_tree(start, split_point, size, chart_span_backtrace_pointers);
    struct  SpanTreeNode* right_child = build_tree(split_point+1, end, size, chart_span_backtrace_pointers);
    root->is_leaf = false;
    root->children.push_back(left_child);
    root->children.push_back(right_child);

    return root;
}

ConTree::~ConTree()
{
    for(auto& p: leaf_input_nodes) {
        if(p) delete p;
    }
    leaf_input_nodes.clear();

    leaf_spantree_nodes.clear();
    phrase_spantree_nodes.clear();

    if(root) {
        delete root;
        root = NULL;
    }

    //cerr<< "destory" << endl;
    if(root != NULL)
    {
        delete root;
        root = NULL;
    }
}

void ConTree::set_spans()
{
    phrase_spantree_nodes.clear();
    this->set_spans(*root);
}

void ConTree::set_spans(struct SpanTreeNode& root)
{
    if(root.phrase_nodes.size() > 0)
    {
        SPAN_KEY key = root.span_left * size() + root.span_right;
        assert(phrase_spantree_nodes.find(key) == phrase_spantree_nodes.end());

        phrase_spantree_nodes[key] = &root;
        /*if(root.phrase_nodes.size() > 3)
        {
            cerr<< "size: " << root.phrase_nodes.size() << endl;
            cerr << root.span_left << " " << root.span_right << " ";
            for(auto p: root.phrase_nodes)
                cerr << p->span_label << " ";
            cerr<<endl;
        }*/
    }

    for(auto& child: root.children)
    {
        set_spans(*child);
    }
}

void ConTree::str(struct SpanTreeNode& root, DictSet& all_dict, vector<string>& strs)
{
    /*if(root.span_left == 0 && root.span_right == (int)size() -1){
        assert(root.phrase_nodes.size() == 1);
    }*/
    for (int phrase_index = (int) root.phrase_nodes.size() - 1; phrase_index >= 0; phrase_index--) {
        PhraseNode *label_node = root.phrase_nodes[phrase_index];
        const string &label_str = all_dict.span_label_dict.dict.convert(label_node->span_label_id);
        if(label_str.back() != '*')
        {
            strs.push_back("(");
            strs.push_back(label_str);
            strs.push_back(" ");
        }
    }
    if (!root.is_leaf) {
        for (auto &child: root.children) {
            str(*child, all_dict, strs);
        }
    } else {
        assert(root.span_right == root.span_left);
        InputNode* input_node = this->leaf_input_nodes[root.span_left];
        strs.push_back("(");
        strs.push_back(input_node->postag);
        strs.push_back(" ");
        if(input_node->word=="(")
            strs.push_back("-LRB-");
        else if(input_node->word==")")
            strs.push_back("-RRB-");
        else
            strs.push_back(input_node->word);
        strs.push_back(")");
    }
    for (int phrase_index = (int) root.phrase_nodes.size() - 1; phrase_index >= 0; phrase_index--) {
        PhraseNode *label_node = root.phrase_nodes[phrase_index];
        const string &label_str = all_dict.span_label_dict.dict.convert(label_node->span_label_id);
        if(label_str.back() != '*')
            strs.push_back(")");
    }

}
#endif //NEURAL_CKY_TREE_H
