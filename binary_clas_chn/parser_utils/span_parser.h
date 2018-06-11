//
// Created by ji_ma on 4/9/17.
//

#ifndef NEURAL_CKY_SPAN_PARSER_H
#define NEURAL_CKY_SPAN_PARSER_H
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
#include "dynet/dict.h"
#include "dynet/training.h"
#include "dynet/nodes.h"
#include "dynet/timing.h"
#include "dynet/lstm.h"
#include "dynet/globals.h"

#include "cl-args.h"
#include "treenode.h"
#include "tree.h"
#include "eval.h"
#include "bitreelstm.h"

using namespace std;
using namespace dynet::expr;
using namespace dynet;

Expression add_or_get_expr(ComputationGraph& cg,
                           LookupParameter& p_table,
                           unordered_map<unsigned, Expression>& cached_exprs,
                           unsigned input_id, const struct Params& params)
{
    Expression input_expr;
    auto iter = cached_exprs.find(input_id);
    if(iter != cached_exprs.end())
    {
        input_expr = iter->second;
    }
    else {
        input_expr = lookup(cg,  p_table, input_id);
        if (!params.eval) {
            input_expr = dropout(input_expr, params.pdrop);
        }
        cached_exprs[input_id] = input_expr;
    }
    return input_expr;
}

template<class Builder>
struct BiLSTMBuilder {
    Parameter p_padding_start;
    Parameter p_padding_end;

    Builder fwd_encoder;
    Builder rev_encoder;

    Expression e_padding_start;
    Expression e_padding_end;

    BiLSTMBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, Model& model):
        fwd_encoder(layers, input_dim, hidden_dim, model),
        rev_encoder(layers, input_dim, hidden_dim, model)
    {
        p_padding_start = model.add_parameters({input_dim});
        p_padding_end = model.add_parameters({input_dim});
    }

    void new_graph(ComputationGraph& cg)
    {
        fwd_encoder.new_graph(cg);
        rev_encoder.new_graph(cg);
        e_padding_start = parameter(cg, p_padding_start);
        e_padding_end = parameter(cg, p_padding_end);
    }

    void build_graph(const vector<Expression>& input_exprs,
                     vector<Expression>& fwd_exprs, vector<Expression>& rev_exprs, bool add_start=false)
    {
        unsigned slen = input_exprs.size();
        if(add_start)
        {
            fwd_exprs.resize(slen+1);
            rev_exprs.resize(slen+1);
        } else {
            fwd_exprs.resize(slen);
            rev_exprs.resize(slen);
        }

        fwd_encoder.start_new_sequence();
        rev_encoder.start_new_sequence();

        fwd_encoder.add_input(e_padding_start);
        if(add_start)
            fwd_exprs[0] = fwd_encoder.back();
        for (unsigned token_index = 0; token_index < slen; ++token_index)
        {
            if(add_start)
                fwd_exprs[token_index+1] = fwd_encoder.add_input(input_exprs[token_index]);
            else
                fwd_exprs[token_index] = fwd_encoder.add_input(input_exprs[token_index]);
        }

        rev_encoder.add_input(e_padding_end);
        if(add_start)
            rev_exprs[slen] = rev_encoder.back();
        for (unsigned token_index = 0; token_index < slen; ++token_index)
        {
            rev_exprs[slen-1u-token_index] = rev_encoder.add_input(input_exprs[slen-1u-token_index]);
        }
    }

    void build_batched_graph(const vector<Expression>& input_exprs,
                             vector<Expression>& fwd_exprs, vector<Expression>& rev_exprs,
                             unsigned bsize)
    {
        unsigned slen = input_exprs.size();
        fwd_exprs.resize(slen);
        rev_exprs.resize(slen);

        fwd_encoder.start_new_sequence();
        rev_encoder.start_new_sequence();

        Expression batched_padding_start = concatenate_to_batch(vector<Expression>(bsize, e_padding_start));
        Expression batched_padding_end = concatenate_to_batch(vector<Expression>(bsize, e_padding_end));

        fwd_encoder.add_input(batched_padding_start);
        for (unsigned token_index = 0; token_index < slen; ++token_index)
        {
            fwd_exprs[token_index] = fwd_encoder.add_input(input_exprs[token_index]);
        }

        rev_encoder.add_input(batched_padding_end);
        for (unsigned token_index = 0; token_index < slen; ++token_index)
        {
            rev_exprs[slen-1u-token_index] = rev_encoder.add_input(input_exprs[slen-1u-token_index]);
        }
    }
};


template <class Builder>
class TokenInputBuilder
{
private:
    LookupParameter p_token_table;
    LookupParameter p_char_table;
    LookupParameter p_postag_table;

    BiLSTMBuilder<Builder>* char_bilstm_builder_ptr;

    Parameter p_input_bias;
    Parameter p_input_char_left;
    Parameter p_input_char_right;


    Expression e_input_bias;
    Expression e_input_char_left;
    Expression e_input_char_right;

    unordered_map<unsigned, Expression> cached_word_exprs;
    unordered_map<unsigned, Expression> cached_char_exprs;
    unordered_map<unsigned, Expression> cached_postag_exprs;

private:
    const struct Params& params;
public:
    TokenInputBuilder(Model& model, const struct Params& _params, Dict& token_dict, unordered_map<string, vector<float> >& pretrained_embeddings):

        params(_params)
    {
        p_token_table = model.add_lookup_parameters(params.TOKEN_VOCAB_SIZE, {params.TOKEN_INPUT_DIM});

        p_postag_table = model.add_lookup_parameters(params.POSTAG_VOCAB_SIZE, {params.POSTAG_INPUT_DIM});

        if(_params.use_char) {
            p_char_table = model.add_lookup_parameters(params.CHAR_VOCAB_SIZE, {params.CHAR_INPUT_DIM});
            p_input_bias = model.add_parameters({params.TOKEN_INPUT_DIM});
            p_input_char_left = model.add_parameters({params.TOKEN_INPUT_DIM, params.CHAR_HIDDEN_DIM});
            p_input_char_right = model.add_parameters({params.TOKEN_INPUT_DIM, params.CHAR_HIDDEN_DIM});
            char_bilstm_builder_ptr = new BiLSTMBuilder<Builder>(_params.CHAR_LSTM_LAYERS, _params.CHAR_INPUT_DIM, _params.CHAR_HIDDEN_DIM, model);
        }
        if(_params.inited_from_pretrained) {
            for (size_t word_index = 0; word_index < params.TOKEN_VOCAB_SIZE; word_index++) {
                const string& word = token_dict.convert(word_index);
                auto word_iter = pretrained_embeddings.find(word);
                if (word_iter != pretrained_embeddings.end()) {
                    p_token_table.initialize(word_index, word_iter->second);
                }  
            }
        }
    }

    ~TokenInputBuilder(){
        if(params.use_char && char_bilstm_builder_ptr != NULL) delete char_bilstm_builder_ptr;
    }


    void new_graph(ComputationGraph& cg)
    {
        if(params.use_char) {
            char_bilstm_builder_ptr->new_graph(cg);
            e_input_bias = parameter(cg, p_input_bias);
            e_input_char_left = parameter(cg, p_input_char_left);
            e_input_char_right = parameter(cg, p_input_char_right);
            cached_char_exprs.clear();
        }
        cached_word_exprs.clear();
        cached_postag_exprs.clear();
    }

    Expression build_graph(ComputationGraph& cg, InputNode* leaf_node)
    {
        Expression combined_token_input;
        auto word_id = leaf_node->word_id;
        auto postag_id = leaf_node->postag_id;

        auto word_expr = add_or_get_expr(cg, p_token_table, cached_word_exprs, word_id, params);
        auto postag_expr = add_or_get_expr(cg, p_postag_table, cached_postag_exprs, postag_id, params);

        if(params.use_char) {
            auto& char_ids = leaf_node->char_ids;
            vector<Expression> char_exprs(char_ids.size());
            for (unsigned char_index = 0; char_index < char_exprs.size(); ++char_index) {
                char_exprs[char_index] = add_or_get_expr(cg, p_char_table, cached_char_exprs, char_ids[char_index], params);
            }
            vector<Expression> combined_input_exprs;
            combined_input_exprs.push_back(e_input_bias);

            vector<Expression> char_fwd_exprs, char_rev_exprs;
            char_bilstm_builder_ptr->build_graph(char_exprs, char_fwd_exprs, char_rev_exprs);

            combined_input_exprs.push_back(e_input_char_left);
            combined_input_exprs.push_back(char_fwd_exprs.back());

            combined_input_exprs.push_back(e_input_char_right);
            combined_input_exprs.push_back(char_rev_exprs[0]);

            Expression nonlinear_input = tanh(affine_transform(combined_input_exprs));
            combined_token_input = word_expr + nonlinear_input;
        }
        else {
            combined_token_input = word_expr;
        }
        return concatenate({combined_token_input, postag_expr});
    }
};


template <class Builder>
struct ConstituentSpanParser {

    BiLSTMBuilder<Builder> word_bilstm_builder;
    BiLSTMBuilder<Builder> l2_word_bilstm_builder;
    TokenInputBuilder<Builder> token_input_builder;
    BinarizedLexTreeLSTMBuilder binary_lextree_lstm_builder;

    Parameter p_left2localh; //span classifier, do we need to include a biaffine score?
    Parameter p_right2localh;
    Parameter p_localh_bias;

    Parameter p_output;
    Parameter p_output_bias;

    Expression i_left2localh;
    Expression i_right2localh;
    Expression i_localh_bias;

    Expression i_output;
    Expression i_output_bias;

    const struct Params &params;
    DictSet& all_dict;

    ConstituentSpanParser(Model &model, const struct Params &_params, DictSet &_all_dict,
                          unordered_map <string, vector<float>> &pretrained_embeddings) :
        word_bilstm_builder(1, _params.TOKEN_INPUT_DIM + _params.POSTAG_INPUT_DIM, _params.HIDDEN_DIM,
                            model),
        l2_word_bilstm_builder(1, 2*_params.HIDDEN_DIM, _params.HIDDEN_DIM,
                               model),
        token_input_builder(model, _params, _all_dict.token_dict.dict, pretrained_embeddings),
        binary_lextree_lstm_builder(model, _params, _all_dict.span_label_dict.dict.convert("</s>")),
        params(_params), all_dict(_all_dict) {

        p_left2localh = model.add_parameters({params.OUTPUT_HIDDEN_DIM, params.HIDDEN_DIM * 2});
        p_right2localh = model.add_parameters({params.OUTPUT_HIDDEN_DIM, params.HIDDEN_DIM * 2});
        p_localh_bias = model.add_parameters({params.OUTPUT_HIDDEN_DIM});

        p_output = model.add_parameters({2, params.OUTPUT_HIDDEN_DIM});
        p_output_bias = model.add_parameters({2});
    }

    void new_graph(ComputationGraph &cg) {
        word_bilstm_builder.new_graph(cg);
        l2_word_bilstm_builder.new_graph(cg);

        token_input_builder.new_graph(cg);
        binary_lextree_lstm_builder.new_graph(cg);

        i_left2localh = parameter(cg, p_left2localh);
        i_right2localh = parameter(cg, p_right2localh);
        i_localh_bias = parameter(cg, p_localh_bias);

        i_output = parameter(cg, p_output);
        i_output_bias = parameter(cg, p_output_bias);
    }

    Expression
    build_graph(ComputationGraph &cg, ConTree *example_tree,
                FMeasure &fmeasure, FMeasure &unlabeld_fmeasure, FMeasure& nontree_unlabeled_fmeasure,
                bool &valid, vector<ConTree*>& pred_trees) {
        const unsigned slen = example_tree->size();

        //if(params.verbose)
        /*if (example_tree->size() > 10) {
            valid = false;
            return input(cg, 0.0f);
        }*/
        if(params.verbose)
            cerr << (*example_tree) << endl;

        //1, get input representation
        vector <Expression> local_input_exprs(slen);
        for (unsigned t = 0; t < slen; ++t) {
            local_input_exprs[t] = token_input_builder.build_graph(cg, example_tree->leaf_input_nodes[t]);
        }
        //2, build lstm
        vector <Expression> fwd_exprs, rev_exprs;
        word_bilstm_builder.build_graph(local_input_exprs, fwd_exprs, rev_exprs);

        vector <Expression> bidir_exprs(slen);
        for (unsigned t = 0; t < slen; ++t) {
            bidir_exprs[t] = concatenate({fwd_exprs[t], rev_exprs[t]});
            if(!params.eval)
                bidir_exprs[t] = dropout(bidir_exprs[t], params.pdrop);
        }

        vector <Expression> l2_fwd_exprs, l2_rev_exprs;
        l2_word_bilstm_builder.build_graph(bidir_exprs, l2_fwd_exprs, l2_rev_exprs);

        vector <Expression> l2_bidir_exprs(slen);
        for (unsigned t = 0; t < slen; ++t) {
            l2_bidir_exprs[t] = concatenate({l2_fwd_exprs[t], l2_rev_exprs[t]});
            Expression l2_bidir_in = (!params.eval && params.ptreedrop> 0)? dropout(l2_bidir_exprs[t], params.ptreedrop) : l2_bidir_exprs[t];
            example_tree->leaf_spantree_nodes[t]->xi = concatenate({l2_bidir_in, local_input_exprs[t]});
        }
        //3, collect softmax errors
        unordered_map <SPAN_KEY, vector<float>> chart_probs;
        unordered_set<int> predicted_spans;

        vector<Expression> left_exprs(((slen -1) * slen) / 2);
        vector<Expression> right_exprs(((slen -1) * slen) / 2);
        vector<unsigned> gold_indices(((slen -1) * slen) / 2);

        unsigned span_index = 0u;
        for (unsigned span_left = 0u; span_left < slen; ++span_left) {
            for (unsigned span_right = span_left+1; span_right < slen; ++span_right) {

                left_exprs[span_index] = l2_bidir_exprs[span_left];
                right_exprs[span_index] = l2_bidir_exprs[span_right];

                SPAN_KEY gold_key = span_left * slen + span_right;

                unsigned gold_index = (example_tree->phrase_spantree_nodes.find(gold_key) ==
                                       example_tree->phrase_spantree_nodes.end()) ? 0u : 1u;

                gold_indices[span_index++] = gold_index;


            }
        }
        assert(span_index == ((slen - 1) * slen) / 2);

        Expression loss;
        if(gold_indices.size() > 0 ) {

            Expression batched_left_expr = concatenate_to_batch(left_exprs);
            Expression batched_right_expr = concatenate_to_batch(right_exprs);

            Expression i_span_h = tanh(affine_transform({i_localh_bias, i_left2localh, batched_left_expr, i_right2localh, batched_right_expr}));

            Expression i_prob_output = affine_transform({i_output_bias, i_output, i_span_h});
            Expression i_final = i_prob_output;
            if(params.eval) {
                cg.incremental_forward(i_final);
                auto i_final_values = as_vector(log_softmax(i_final).value());
                span_index = 0;
                for (unsigned span_left = 0u; span_left < slen; ++span_left) {
                    for (unsigned span_right = span_left + 1; span_right < slen; ++span_right) {

                        SPAN_KEY gold_key = span_left * slen + span_right;

                        vector<float> tag_probs(i_final_values.begin()+span_index*2, i_final_values.begin()+(span_index+1)*2);

                        chart_probs[gold_key] = tag_probs; //log probs

                        if (tag_probs[0] < tag_probs[1]) {
                            //cerr<<"tag_probs: " << tag_probs[1] << endl;
                            predicted_spans.insert(gold_key);
                        }
                        /*else{
                        cerr<<"tag_probs: " << tag_probs[0] << endl;
                        }*/
                        span_index++;
                    }
                }
                assert(span_index == ((slen - 1) * slen) / 2);
            }
            Expression errs = pickneglogsoftmax(i_final, gold_indices);
            loss = sum_batches(errs);
        }

        if(!params.eval)
        {
            Expression label_loss = binary_lextree_lstm_builder.build_graph(cg, example_tree);
            if(gold_indices.size() > 0)
                loss = loss + label_loss;
            else
                loss = label_loss;
        }
        else {
            ConTree* pred_tree = binary_cky_parsing(example_tree, chart_probs);
            for (unsigned t = 0; t < slen; ++t) {
                pred_tree->leaf_spantree_nodes[t]->xi = example_tree->leaf_spantree_nodes[t]->xi;
            }

            Expression pred_loss = binary_lextree_lstm_builder.generate_labels(cg, pred_tree);
            if(gold_indices.size() > 0)
                loss  = loss + pred_loss;
            else
                loss = pred_loss;

            if(params.verbose)
            {
                cerr << endl << "pred_tree: " << endl;
                cerr << (*pred_tree) << endl;
            }

            {
                fmeasure.num_examples++;
                unlabeld_fmeasure.num_examples++;

                nontree_unlabeled_fmeasure.num_examples++;

                for(auto& key: predicted_spans)
                {
                    int pred_left = key / slen;
                    int pred_right = key % slen;
                    if(params.verbose)
                    {
                        cerr<<"\tnontree [" << key / slen  << ", " << key % slen << "]" << endl;
                    }
                    if(pred_left != pred_right) {
                        nontree_unlabeled_fmeasure.num_prediction++;
                        if (example_tree->phrase_spantree_nodes.find(key) !=
                                example_tree->phrase_spantree_nodes.end()) {
                            nontree_unlabeled_fmeasure.num_common++;
                        }
                    }
                }

                for(auto& item: example_tree->phrase_spantree_nodes)
                {
                    int gold_left = item.first / slen;
                    int gold_right = item.first % slen;

                    unlabeld_fmeasure.num_total++;
                    if(gold_left != gold_right)
                        nontree_unlabeled_fmeasure.num_total++;

                    for(auto& p: item.second->phrase_nodes) {
                        const string& label_str = p->span_label;
                        if(label_str.back() != '*') {
                            //if(gold_left == 0 && gold_right == (int)slen-1 && p->span_label_id == all_dict.span_label_dict.dict.convert("S"))
                            //    continue;
                            fmeasure.num_total ++;
                            if(params.verbose)
                            {
                                cerr << "\tg[" << gold_left << ", " << gold_right << "] ";
                                cerr<< label_str.size() << "\t" << label_str << endl;
                            }
                        }
                    }
                }
                if(params.verbose)
                    cerr << "total: " << fmeasure.num_total << endl;

                for(auto& item: pred_tree->phrase_spantree_nodes)
                {
                    int pred_left = item.first / slen;
                    int pred_right = item.first % slen;

                    unlabeld_fmeasure.num_prediction++;

                    auto gold_key_iter = example_tree->phrase_spantree_nodes.find(item.first);

                    unlabeld_fmeasure.num_common += (gold_key_iter != example_tree->phrase_spantree_nodes.end());

                    for(auto& p: item.second->phrase_nodes) {
                        const string& label_str = all_dict.span_label_dict.dict.convert(p->span_label_id);
                        if(label_str.back() != '*') {
                            //if(pred_left == 0 && pred_right == (int)slen-1 && p->span_label_id == all_dict.span_label_dict.dict.convert("S"))
                            //    continue;
                            fmeasure.num_prediction ++;
                            if(params.verbose)
                            {
                                cerr << "\tp[" << pred_left << ", " << pred_right << "] ";
                                cerr<< label_str.size() << "\t" << label_str << endl;
                            }

                            if(gold_key_iter != example_tree->phrase_spantree_nodes.end())
                            {
                                auto& gold_phrase_nodes = gold_key_iter->second->phrase_nodes;
                                for(auto& g: gold_phrase_nodes) {
                                    if(p->span_label_id == g->span_label_id)
                                    {
                                        fmeasure.num_common++;
                                        //cerr<<"common" << endl;
                                        //exit(0);
                                    }
                                }
                            }
                        }
                    }
                }
                if(params.verbose)
                {
                    cerr << "prediction: " << fmeasure.num_prediction << endl;
                    cerr << "num_common: " << fmeasure.num_common << endl;
                }
            }

            //Expression loss = span_loss;
            pred_trees.push_back(pred_tree);
            if(params.verbose)
                cerr<<"pred_size: " << pred_trees.size() << endl;
        }
        valid = true;
        //if(example_tree->size() < 10)
        //exit(0);
        return loss;
    }

    ConTree* binary_cky_parsing(ConTree *example_tree, const unordered_map <SPAN_KEY, vector<float>> &chart_probs) {
        if(params.verbose)
        {
            cerr << "[tlog] decoding using cky" << endl;
            for (auto &item: chart_probs) {
                int span_left = item.first / example_tree->size();
                int span_right = item.first % example_tree->size();
                {
                    cerr << "\t[" << span_left << ", " << span_right << "] ";
                    for (auto &p: item.second)
                        cerr << p << " ";
                    cerr << endl;
                }
            }
        }

        unordered_map<SPAN_KEY, float> chart_span_probs;
        unordered_map<SPAN_KEY, int> chart_span_backtrace_pointers;

        for (auto &key: chart_probs) {
            chart_span_probs[key.first] = key.second[1]; //only keeping has_span probability
        }

        for (int l = 2; l <= (int) example_tree->size(); l++) {
            for (int span_start = 0; span_start + l <= (int) example_tree->size(); span_start++) {
                int span_end = span_start + l - 1; // [0, 1] for 2
                if(params.verbose)
                    cerr << "[tlog] decoding span [ " << span_start << ", " << span_end << " ]" << endl;

                float max_prob = 0.0f;
                int max_split_index = -1;

                for (int k = span_start; k < span_end; k++) {
                    //[span_start, k] [k+1, span_end]

                    SPAN_KEY left_key = span_start * example_tree->size() + k;
                    SPAN_KEY right_key = (k + 1) * example_tree->size() + span_end;

                    //1 for leaf nodes
                    float left_prob = 0.0f, right_prob = 0.0f;

                    if (span_start < k) {
                        auto left_span_iter = chart_span_probs.find(left_key);
                        left_prob = left_span_iter->second;
                        assert(left_span_iter != chart_span_probs.end());
                    }
                    if (k + 1 < span_end) {
                        auto right_span_iter = chart_span_probs.find(right_key);
                        assert(right_span_iter != chart_span_probs.end());
                        right_prob = right_span_iter->second;
                    }

                    float candidate_prob = left_prob + right_prob;

                    if (max_split_index == -1 || max_prob < candidate_prob) {
                        max_split_index = k;
                        max_prob = candidate_prob;
                    }
                    if(params.verbose)
                        cerr<<"\t\t[debug] " << k << "\t" << candidate_prob << endl;
                }
                if(params.verbose)
                    cerr << "[tlog] split_point: " << max_split_index << endl;
                assert(max_split_index != -1);
                SPAN_KEY whole_span_key = span_start * example_tree->size() + span_end;
                chart_span_probs[whole_span_key] += max_prob;
                chart_span_backtrace_pointers[whole_span_key] = max_split_index;
            }
        }
        if(params.verbose)
            display(0, example_tree->size()-1, example_tree->size(), chart_span_backtrace_pointers);

        ConTree* pred_tree = new ConTree(example_tree->size(), chart_span_backtrace_pointers);
        return pred_tree;
    }

    void display(int start, int end, int size, const unordered_map<SPAN_KEY, int> &chart_span_backtrace_pointers)
    {
        if(params.verbose)
            cerr << "[ " << start << ", " << end << " ] " ;
        if(start >= end)
            return;
        SPAN_KEY  span_key = start * size + end;
        int split_point =  chart_span_backtrace_pointers.find(span_key)->second;
        if(params.verbose)
            cerr << split_point << endl;

        display(start, split_point, size, chart_span_backtrace_pointers);
        display(split_point+1, end, size, chart_span_backtrace_pointers);
    }

private:

};

#endif //NEURAL_CKY_SPAN_PARSER_H
