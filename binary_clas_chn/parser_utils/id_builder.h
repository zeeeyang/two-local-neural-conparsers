//
// Created by ji_ma on 5/15/17.
//

#ifndef NEURAL_CKY_ID_BUILDER_H
#define NEURAL_CKY_ID_BUILDER_H

#include "dynet/dict.h"
#include "dynet/training.h"
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/timing.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "dynet/globals.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>


#include "../parser_utils/cl-args.h"
#include "../parser_utils/eval.h"
#include "../parser_utils/tree.h"

using namespace std;

class IdBuilder {
protected:
    unordered_map<string, vector<float> > pretrained_embeddings;
    unordered_map<string, string> unk_map;

    vector<float> averaged_vec;
    unordered_map<string, int> word_counts;
    DictSet all_dict;
    bool add_pretrained_types_to_dict;
    bool add_unk_types_to_dict;
public:
    //including everything necessary for making id
    IdBuilder(): add_pretrained_types_to_dict(true), add_unk_types_to_dict(false) {

    }

    inline unordered_map<string, vector<float> >& get_pretrained_embeddings() {
        return pretrained_embeddings;
    }

    inline DictSet& get_all_dict() {
        return all_dict;
    }

    inline vector<float>& get_averaged_vec() {
        return averaged_vec;
    }

    inline unordered_map<string, string>& get_unk_map() {
        return unk_map;
    }

    inline bool can_add_pretrained_types_to_dict() {
        return add_pretrained_types_to_dict;
    }

    inline bool can_add_unk_types_to_dict() {
        return add_unk_types_to_dict;
    }

    virtual void build_id_for_dev_examples(vector<ConTree*>& dev_trees)
    {
        for(size_t example_index = 0u; example_index < dev_trees.size(); example_index++) {

            auto tree_ptr = dev_trees[example_index];

            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];

                leaf_node->word_id = all_dict.token_dict.dict.convert(leaf_node->word);

                if(leaf_node->word_id == all_dict.token_dict.kUNK) {
                    leaf_node->word_id = all_dict.token_dict.dict.convert(leaf_node->lowercased_word);
                }

                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
    }

    virtual void build_id_for_examples(vector<ConTree*>& training_trees, vector<ConTree*>& dev_trees, vector<ConTree*>& test_trees)
    {
        unordered_map<string, int> char_counts;

        unordered_map<string, int> span_label_counts;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++)
        {
            auto tree_ptr = training_trees[example_index];
            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++)
            {
                word_counts[tree_ptr->leaf_input_nodes[word_index]->word]++;

                for(auto& character: tree_ptr->leaf_input_nodes[word_index]->chars)
                    char_counts[character]++;
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes)
            {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    span_label_counts[treenode->span_label]++;
                }
            }
        }
        for(auto& span_label_item: span_label_counts)
            assert(span_label_item.second >= 1);

        //Now we have the counts
        //train
        int token_cut_off = 1;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++) {

            auto tree_ptr = training_trees[example_index];
            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];

                if(token_cut_off >=  word_counts[leaf_node->word])
                {
                    leaf_node->is_single = false;
                    if(all_dict.token_dict.dict.contains(leaf_node->word))
                        leaf_node->word_id = all_dict.token_dict.dict.convert(leaf_node->word);
                    else {
                        bool lower_case_found = (pretrained_embeddings.find(leaf_node->lowercased_word) !=
                                                 pretrained_embeddings.end() || all_dict.token_dict.dict.contains(leaf_node->lowercased_word));
                        if (lower_case_found) {
                            leaf_node->word_id = all_dict.token_dict.dict.convert(leaf_node->lowercased_word);
                        } else {
                            leaf_node->word_id = all_dict.token_dict.kUNK;
                        }
                    }
                }
                else {
                    leaf_node->word_id = all_dict.token_dict.dict.convert(leaf_node->word);
                }

                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    if( token_cut_off <  char_counts[char_i] )
                        leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                    else
                        leaf_node->char_ids[char_index] = all_dict.char_dict.kUNK;
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
        all_dict.span_label_dict.dict.convert("</s>"); //end label
        all_dict.freeze();

        cerr << "[tlog] end_label: " << all_dict.span_label_dict.dict.convert("</s>") << endl;
        for(unsigned label_index = 0u; label_index < all_dict.span_label_dict.dict.size(); label_index++)
            cerr << "[tlog] label_index: " << label_index << "\t" << all_dict.span_label_dict.dict.convert(label_index) << endl;

        all_dict.token_dict.set_unk();
        all_dict.char_dict.set_unk();
        //POS tag
        all_dict.postag_dict.set_unk(); // For SYM, strange
        //dev
        build_id_for_dev_examples(dev_trees);
        //test
        build_id_for_dev_examples(test_trees);
    }

    virtual void stochastic_replace(ConTree* tree, Params& params)
    {
        return;
    }
};

class NoPretrainedUnkMapJamesIdBuilder: public IdBuilder {

public:
    NoPretrainedUnkMapJamesIdBuilder(): IdBuilder() {
        this->add_pretrained_types_to_dict = false;
        this->add_unk_types_to_dict = true;
    }

    void build_id_for_dev_examples(vector<ConTree*>& dev_trees)
    {
        for(size_t example_index = 0u; example_index < dev_trees.size(); example_index++) {

            auto tree_ptr = dev_trees[example_index];

            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];

                leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->word);

                if(leaf_node->original_word_id == all_dict.token_dict.kUNK) {
                    leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->lowercased_word);
                }

                leaf_node->unk_id = all_dict.token_dict.kUNK;
                auto unk_iter = this->unk_map.find(leaf_node->word);
                if(unk_iter != this->unk_map.end())
                {
                    leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                }

                if(leaf_node->unk_id == all_dict.token_dict.kUNK) {
                    unk_iter = this->unk_map.find(leaf_node->lowercased_word);
                    if(unk_iter != this->unk_map.end())
                    {
                        leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                    }
                }

                if(leaf_node->original_word_id == all_dict.token_dict.kUNK) {
                    leaf_node->word_id = leaf_node->unk_id;
                } else {
                    leaf_node->word_id = leaf_node->original_word_id;
                }
                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
    }

    void build_id_for_examples(vector<ConTree*>& training_trees, vector<ConTree*>& dev_trees, vector<ConTree*>& test_trees)
    {
        unordered_map<string, int> char_counts;

        unordered_map<string, int> span_label_counts;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++)
        {
            auto tree_ptr = training_trees[example_index];
            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++)
            {
                word_counts[tree_ptr->leaf_input_nodes[word_index]->word]++;

                for(auto& character: tree_ptr->leaf_input_nodes[word_index]->chars)
                    char_counts[character]++;
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes)
            {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    span_label_counts[treenode->span_label]++;
                }
            }
        }
        for(auto& span_label_item: span_label_counts)
            assert(span_label_item.second >= 1);

        //Now we have the counts
        //train
        int token_cut_off = 1;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++) {

            auto tree_ptr = training_trees[example_index];

            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];

                leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->word);

                leaf_node->word_id = leaf_node->original_word_id;

                leaf_node->unk_id = all_dict.token_dict.kUNK;
                auto unk_iter = this->unk_map.find(leaf_node->word);
                if(unk_iter != this->unk_map.end())
                {
                    leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                }

                if(leaf_node->unk_id == all_dict.token_dict.kUNK) {
                    unk_iter = this->unk_map.find(leaf_node->lowercased_word);
                    if(unk_iter != this->unk_map.end())
                    {
                        leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                    }
                }

                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    if( token_cut_off <  char_counts[char_i] )
                        leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                    else
                        leaf_node->char_ids[char_index] = all_dict.char_dict.kUNK;
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
        all_dict.span_label_dict.dict.convert("</s>"); //end label
        all_dict.freeze();

        cerr << "[tlog] end_label: " << all_dict.span_label_dict.dict.convert("</s>") << endl;
        for(unsigned label_index = 0u; label_index < all_dict.span_label_dict.dict.size(); label_index++)
            cerr << "[tlog] label_index: " << label_index << "\t" << all_dict.span_label_dict.dict.convert(label_index) << endl;

        all_dict.token_dict.set_unk();
        all_dict.char_dict.set_unk();
        //POS tag
        all_dict.postag_dict.set_unk(); // For SYM, strange
        //dev
        build_id_for_dev_examples(dev_trees);
        //test
        build_id_for_dev_examples(test_trees);
    }

    void stochastic_replace(ConTree* tree, Params& params)
    {
        assert(!params.eval);
        for(unsigned word_index = 0u; word_index < tree->size(); word_index++) {

            auto& leaf_node = tree->leaf_input_nodes[word_index];

            float word_count = word_counts[leaf_node->word];

            float drop_prob =  params.UNK_Z / (params.UNK_Z + word_count);

            if(dynet::rand01() < drop_prob) {
                leaf_node->word_id = leaf_node->unk_id;
            } else {
                leaf_node->word_id = leaf_node->original_word_id;
            }
        }
    }
};


class NtParserIdBuilder: public IdBuilder {

public:

    unordered_set<unsigned> singletons;

    NtParserIdBuilder(): IdBuilder() {
        this->add_pretrained_types_to_dict = true;
        this->add_unk_types_to_dict = true;
    }

    void build_id_for_dev_examples(vector<ConTree*>& dev_trees)
    {
        for(size_t example_index = 0u; example_index < dev_trees.size(); example_index++) {

            auto tree_ptr = dev_trees[example_index];

            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];

                leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->word);

                if(leaf_node->original_word_id == all_dict.token_dict.kUNK) {
                    leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->lowercased_word);
                }

                leaf_node->unk_id = all_dict.token_dict.kUNK;
                auto unk_iter = this->unk_map.find(leaf_node->word);
                if(unk_iter != this->unk_map.end())
                {
                    leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                }

                if(leaf_node->unk_id == all_dict.token_dict.kUNK) {
                    unk_iter = this->unk_map.find(leaf_node->lowercased_word);
                    if(unk_iter != this->unk_map.end())
                    {
                        leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                    }
                }

                if(leaf_node->original_word_id == all_dict.token_dict.kUNK) {
                    leaf_node->word_id = leaf_node->unk_id;
                } else {
                    leaf_node->word_id = leaf_node->original_word_id;
                }
                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
    }

    void build_id_for_examples(vector<ConTree*>& training_trees, vector<ConTree*>& dev_trees, vector<ConTree*>& test_trees)
    {
        unordered_map<string, int> char_counts;

        unordered_map<string, int> span_label_counts;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++)
        {
            auto tree_ptr = training_trees[example_index];
            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++)
            {
                word_counts[tree_ptr->leaf_input_nodes[word_index]->word]++;

                for(auto& character: tree_ptr->leaf_input_nodes[word_index]->chars)
                    char_counts[character]++;
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes)
            {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    span_label_counts[treenode->span_label]++;
                }
            }
        }
        for(auto& span_label_item: span_label_counts)
            assert(span_label_item.second >= 1);


        //Now we have the counts
        //train
        int token_cut_off = 1;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++) {

            auto tree_ptr = training_trees[example_index];

            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];
                //including all words in the training set to the vocabulary
                leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->word);

                leaf_node->word_id = leaf_node->original_word_id;

                leaf_node->unk_id = all_dict.token_dict.kUNK;
                auto unk_iter = this->unk_map.find(leaf_node->word);
                if(unk_iter != this->unk_map.end())
                {
                    leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                }

                if(leaf_node->unk_id == all_dict.token_dict.kUNK) {
                    unk_iter = this->unk_map.find(leaf_node->lowercased_word);
                    if(unk_iter != this->unk_map.end())
                    {
                        leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                    }
                }

                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    if( token_cut_off <  char_counts[char_i] )
                        leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                    else
                        leaf_node->char_ids[char_index] = all_dict.char_dict.kUNK;
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
        all_dict.span_label_dict.dict.convert("</s>"); //end label
        all_dict.freeze();

        for(auto& word_item: word_counts)
            if(word_item.second == 1)
                singletons.insert(all_dict.token_dict.dict.convert(word_item.first));
        cerr <<"[tlog] #num of singletons: " << singletons.size() << endl;
        cerr << "[tlog] end_label: " << all_dict.span_label_dict.dict.convert("</s>") << endl;
        for(unsigned label_index = 0u; label_index < all_dict.span_label_dict.dict.size(); label_index++)
            cerr << "[tlog] label_index: " << label_index << "\t" << all_dict.span_label_dict.dict.convert(label_index) << endl;

        all_dict.token_dict.set_unk();
        all_dict.char_dict.set_unk();
        //POS tag
        all_dict.postag_dict.set_unk(); // For SYM, strange
        //dev
        build_id_for_dev_examples(dev_trees);
        //test
        build_id_for_dev_examples(test_trees);
    }

    void stochastic_replace(ConTree* tree, Params& params)
    {
        assert(!params.eval);
        for(unsigned word_index = 0u; word_index < tree->size(); word_index++) {

            auto& leaf_node = tree->leaf_input_nodes[word_index];

            float drop_prob =  0.5f;

            if(singletons.count(leaf_node->original_word_id) && dynet::rand01() < 0.5f) {
                leaf_node->word_id = leaf_node->unk_id;
            } else {
                leaf_node->word_id = leaf_node->original_word_id;
            }
        }
    }
};


class UnkMapIdBuilder: public IdBuilder {
//myself builder + unk map
public:
    UnkMapIdBuilder(): IdBuilder() {
        this->add_pretrained_types_to_dict = true;
        this->add_unk_types_to_dict = true;
    }

    void build_id_for_dev_examples(vector<ConTree*>& dev_trees)
    {
        for(size_t example_index = 0u; example_index < dev_trees.size(); example_index++) {

            auto tree_ptr = dev_trees[example_index];

            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];

                leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->word);

                if(leaf_node->original_word_id == all_dict.token_dict.kUNK) {
                    leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->lowercased_word);
                }

                leaf_node->unk_id = all_dict.token_dict.kUNK;
                auto unk_iter = this->unk_map.find(leaf_node->word);
                if(unk_iter != this->unk_map.end())
                {
                    leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                }

                if(leaf_node->unk_id == all_dict.token_dict.kUNK) {
                    unk_iter = this->unk_map.find(leaf_node->lowercased_word);
                    if(unk_iter != this->unk_map.end())
                    {
                        leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                    }
                }

                if(leaf_node->original_word_id == all_dict.token_dict.kUNK) {
                    leaf_node->word_id = leaf_node->unk_id;
                } else {
                    leaf_node->word_id = leaf_node->original_word_id;
                }
                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
    }

    void build_id_for_examples(vector<ConTree*>& training_trees, vector<ConTree*>& dev_trees, vector<ConTree*>& test_trees)
    {
        unordered_map<string, int> char_counts;

        unordered_map<string, int> span_label_counts;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++)
        {
            auto tree_ptr = training_trees[example_index];
            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++)
            {
                word_counts[tree_ptr->leaf_input_nodes[word_index]->word]++;

                for(auto& character: tree_ptr->leaf_input_nodes[word_index]->chars)
                    char_counts[character]++;
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes)
            {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    span_label_counts[treenode->span_label]++;
                }
            }
        }
        for(auto& span_label_item: span_label_counts)
            assert(span_label_item.second >= 1);

        //Now we have the counts
        //train
        int token_cut_off = 1;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++) {

            auto tree_ptr = training_trees[example_index];

            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];

                if(token_cut_off >=  word_counts[leaf_node->word])
                {

                    leaf_node->is_single = false;

                    if(all_dict.token_dict.dict.contains(leaf_node->word))
                        leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->word);
                    else {
                        bool lower_case_found = (pretrained_embeddings.find(leaf_node->lowercased_word) !=
                                                 pretrained_embeddings.end() || all_dict.token_dict.dict.contains(leaf_node->lowercased_word));
                        if (lower_case_found) {
                            leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->lowercased_word);
                        } else {
                            leaf_node->unk_id = all_dict.token_dict.kUNK;
                            auto unk_iter = this->unk_map.find(leaf_node->word);
                            if(unk_iter != this->unk_map.end())
                            {
                                leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                            }

                            if(leaf_node->unk_id == all_dict.token_dict.kUNK) {
                                unk_iter = this->unk_map.find(leaf_node->lowercased_word);
                                if(unk_iter != this->unk_map.end())
                                {
                                    leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                                }
                            }
                            leaf_node->original_word_id = leaf_node->unk_id;
                        }
                    }
                }
                else {
                    leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->word);
                }
                leaf_node->word_id = leaf_node->original_word_id;

                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    if( token_cut_off <  char_counts[char_i] )
                        leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                    else
                        leaf_node->char_ids[char_index] = all_dict.char_dict.kUNK;
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
        all_dict.span_label_dict.dict.convert("</s>"); //end label
        all_dict.freeze();

        cerr << "[tlog] end_label: " << all_dict.span_label_dict.dict.convert("</s>") << endl;
        for(unsigned label_index = 0u; label_index < all_dict.span_label_dict.dict.size(); label_index++)
            cerr << "[tlog] label_index: " << label_index << "\t" << all_dict.span_label_dict.dict.convert(label_index) << endl;

        all_dict.token_dict.set_unk();
        all_dict.char_dict.set_unk();
        //POS tag
        all_dict.postag_dict.set_unk(); // For SYM, strange
        //dev
        build_id_for_dev_examples(dev_trees);
        //test
        build_id_for_dev_examples(test_trees);
    }

    void stochastic_replace(ConTree* tree, Params& params)
    {

    }
};


class PretrainedUnkMapJamesIdBuilder: public IdBuilder {

public:
    PretrainedUnkMapJamesIdBuilder(): IdBuilder() {
        this->add_pretrained_types_to_dict = true;
        this->add_unk_types_to_dict = true;
    }

    void build_id_for_dev_examples(vector<ConTree*>& dev_trees)
    {
        for(size_t example_index = 0u; example_index < dev_trees.size(); example_index++) {

            auto tree_ptr = dev_trees[example_index];

            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];

                leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->word);

                if(leaf_node->original_word_id == all_dict.token_dict.kUNK) {
                    leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->lowercased_word);
                }

                leaf_node->unk_id = all_dict.token_dict.kUNK;
                auto unk_iter = this->unk_map.find(leaf_node->word);
                if(unk_iter != this->unk_map.end())
                {
                    leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                }

                if(leaf_node->unk_id == all_dict.token_dict.kUNK) {
                    unk_iter = this->unk_map.find(leaf_node->lowercased_word);
                    if(unk_iter != this->unk_map.end())
                    {
                        leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                    }
                }

                if(leaf_node->original_word_id == all_dict.token_dict.kUNK) {
                    leaf_node->word_id = leaf_node->unk_id;
                } else {
                    leaf_node->word_id = leaf_node->original_word_id;
                }
                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
    }

    void build_id_for_examples(vector<ConTree*>& training_trees, vector<ConTree*>& dev_trees, vector<ConTree*>& test_trees)
    {
        unordered_map<string, int> char_counts;

        unordered_map<string, int> span_label_counts;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++)
        {
            auto tree_ptr = training_trees[example_index];
            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++)
            {
                word_counts[tree_ptr->leaf_input_nodes[word_index]->word]++;

                for(auto& character: tree_ptr->leaf_input_nodes[word_index]->chars)
                    char_counts[character]++;
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes)
            {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    span_label_counts[treenode->span_label]++;
                }
            }
        }
        for(auto& span_label_item: span_label_counts)
            assert(span_label_item.second >= 1);

        //Now we have the counts
        //train
        int token_cut_off = 1;
        for(size_t example_index = 0u; example_index < training_trees.size(); example_index++) {

            auto tree_ptr = training_trees[example_index];

            for(unsigned word_index = 0u; word_index < tree_ptr->size(); word_index++) {

                auto& leaf_node = tree_ptr->leaf_input_nodes[word_index];

                leaf_node->original_word_id = all_dict.token_dict.dict.convert(leaf_node->word);

                leaf_node->word_id = leaf_node->original_word_id;

                leaf_node->unk_id = all_dict.token_dict.kUNK;
                auto unk_iter = this->unk_map.find(leaf_node->word);
                if(unk_iter != this->unk_map.end())
                {
                    leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                }

                if(leaf_node->unk_id == all_dict.token_dict.kUNK) {
                    unk_iter = this->unk_map.find(leaf_node->lowercased_word);
                    if(unk_iter != this->unk_map.end())
                    {
                        leaf_node->unk_id = all_dict.token_dict.dict.convert(unk_iter->second);
                    }
                }

                leaf_node->char_ids.resize(leaf_node->chars.size());
                for(size_t char_index = 0u; char_index <  leaf_node->chars.size(); char_index++) {
                    auto& char_i = leaf_node->chars[char_index];
                    if( token_cut_off <  char_counts[char_i] )
                        leaf_node->char_ids[char_index] = all_dict.char_dict.dict.convert( char_i );
                    else
                        leaf_node->char_ids[char_index] = all_dict.char_dict.kUNK;
                }

                leaf_node->postag_id = all_dict.postag_dict.dict.convert(leaf_node->postag);
            }
            for(auto& tree_item: tree_ptr->phrase_spantree_nodes) {
                for(auto& treenode: tree_item.second->phrase_nodes) {
                    treenode->span_label_id = all_dict.span_label_dict.dict.convert(treenode->span_label);
                }
            }
        }
        all_dict.span_label_dict.dict.convert("</s>"); //end label
        all_dict.freeze();

        cerr << "[tlog] end_label: " << all_dict.span_label_dict.dict.convert("</s>") << endl;
        for(unsigned label_index = 0u; label_index < all_dict.span_label_dict.dict.size(); label_index++)
            cerr << "[tlog] label_index: " << label_index << "\t" << all_dict.span_label_dict.dict.convert(label_index) << endl;

        all_dict.token_dict.set_unk();
        all_dict.char_dict.set_unk();
        //POS tag
        all_dict.postag_dict.set_unk(); // For SYM, strange
        //dev
        build_id_for_dev_examples(dev_trees);
        //test
        build_id_for_dev_examples(test_trees);
    }

    void stochastic_replace(ConTree* tree, Params& params)
    {
        assert(!params.eval);
        for(unsigned word_index = 0u; word_index < tree->size(); word_index++) {

            auto& leaf_node = tree->leaf_input_nodes[word_index];

            float word_count = word_counts[leaf_node->word];

            float drop_prob =  params.UNK_Z / (params.UNK_Z + word_count);

            if(dynet::rand01() < drop_prob) {
                leaf_node->word_id = leaf_node->unk_id;
            } else {
                leaf_node->word_id = leaf_node->original_word_id;
            }
        }
    }
};


IdBuilder* create_id_builder(Params& params)
{
    IdBuilder* id_builder = NULL;
    if(params.idbuilder == "default") {
        id_builder =  new IdBuilder;
    } else if(params.idbuilder == "unkid") {
        id_builder =  new UnkMapIdBuilder;
    }  else if(params.idbuilder == "james") {
        id_builder = new NoPretrainedUnkMapJamesIdBuilder;
    } else if(params.idbuilder == "pretrainedjames") { //pretrained
        id_builder = new PretrainedUnkMapJamesIdBuilder;
    } else if(params.idbuilder == "ntparser") {
        id_builder = new NtParserIdBuilder;
    } else {
        cerr<< "You must specify one Idbuilder: default|unkid|james|pretrainedjames|ntparser" << endl;
        exit(0);
    }
    return id_builder;
}

#endif //NEURAL_CKY_ID_BUILDER_H
