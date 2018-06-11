/*************************************************************************
	> File Name: bitreelstm.h
	> Author:
	> Mail:
	> Created Time: Wed 20 Jan 2016 10:15:40 PM SGT
 ************************************************************************/

#ifndef NEURAL_CKY_SPAN_BIN_LEX_TREE_LSTM_H
#define NEURAL_CKY_SPAN_BIN_LEX_TREE_LSTM_H
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
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/timing.h"
#include "dynet/dict.h"

using namespace std;
using namespace dynet::expr;
using namespace dynet;

#include "tree.h"
#include "treenode.h"
#include "cl-args.h"

/*
 * parameters of lstm
 */
class BinarizedLexTreeLSTMBuilder
{
public:

    BinarizedLexTreeLSTMBuilder() = default;

    explicit BinarizedLexTreeLSTMBuilder(Model& model, const struct Params& _params, unsigned _LABEL_END);

    Expression build_graph(ComputationGraph& cg, ConTree* gold_tree);

    void new_graph(ComputationGraph& cg);

    Expression generate_labels(ComputationGraph& cg, ConTree* pred_tree);
private:


    void build_bottomup_graph(ComputationGraph& cg, struct SpanTreeNode& tree);

    Expression get_label_loss(ComputationGraph& cg, struct SpanTreeNode& root);

    void get_label_loss(ComputationGraph& cg, struct SpanTreeNode& root, vector<Expression>& all_h,
                        vector<unsigned>& gold_labels);

    Expression generate_labels(ComputationGraph& cg, struct SpanTreeNode& root);

    void generate_labels(ComputationGraph& cg, struct SpanTreeNode& root, vector<Expression>& allnodes);

    vector<Parameter> params;
    vector<Expression>  param_exprs;

    const struct Params& config;

    unsigned LABEL_END;

    LSTMBuilder label_decoder;

    LookupParameter p_label_table;
};


enum { BX2I,  BLH2I, BLC2I, BRH2I, BRC2I, BBI,  //i
       BX2LF, BLH2LF, BLC2LF, BRH2LF, BRC2LF, BBLF, //l_f
       BX2RF, BLH2RF, BLC2RF, BRH2RF, BRC2RF, BBRF, //r_f
       BX2G,  BLH2G, BRH2G, BBG, //g
       BX2O,  BLH2O, BRH2O, BC2O, BBO,// o

       AVG_L, AVG_R, AVG_LH, AVG_RH, AVG_LC, AVG_RC, AVG_B,

       LABEL_START,

       BU2T, TB,
       T2L, LB
     };

BinarizedLexTreeLSTMBuilder::BinarizedLexTreeLSTMBuilder(Model& model,
        const struct Params& _params,
        unsigned _LABEL_END): config(_params), LABEL_END(_LABEL_END),
    label_decoder(_params.LABEL_LSTM_LAYERS, _params.HIDDEN_DIM + _params.LABEL_INPUT_DIM, _params.HIDDEN_DIM, model)
{
    unsigned layer_input_dim = config.TOKEN_INPUT_DIM + 2*config.HIDDEN_DIM + config.POSTAG_INPUT_DIM;
    unsigned hidden_dim = config.HIDDEN_DIM;

    cerr<< "span_size: " << config.SPAN_LABEL_SIZE << endl;
    cerr<< "span_label input dim: " << config.LABEL_INPUT_DIM << endl;
    //exit(0);
    p_label_table = model.add_lookup_parameters(config.SPAN_LABEL_SIZE, {config.LABEL_INPUT_DIM});

    auto& ps = params;
    // i
    Parameter p_bx2i = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_blh2i = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_blc2i = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_brh2i = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_brc2i = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bbi = model.add_parameters({hidden_dim});
    ps.push_back(p_bx2i);
    ps.push_back(p_blh2i);
    ps.push_back(p_blc2i);
    ps.push_back(p_brh2i);
    ps.push_back(p_brc2i);
    ps.push_back(p_bbi);

    Parameter p_bx2lf = model.add_parameters({hidden_dim, layer_input_dim});
    ps.push_back(p_bx2lf);
    Parameter p_blh2lf = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_blc2lf = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_brh2lf = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_brc2lf = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bblf = model.add_parameters({hidden_dim});
    ps.push_back(p_blh2lf);
    ps.push_back(p_blc2lf);
    ps.push_back(p_brh2lf);
    ps.push_back(p_brc2lf);
    ps.push_back(p_bblf);

    Parameter p_bx2rf = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_blh2rf = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_blc2rf = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_brh2rf = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_brc2rf = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bbrf = model.add_parameters({hidden_dim});
    ps.push_back(p_bx2rf);
    ps.push_back(p_blh2rf);
    ps.push_back(p_blc2rf);
    ps.push_back(p_brh2rf);
    ps.push_back(p_brc2rf);
    ps.push_back(p_bbrf);

    //BX2G, BLH2G, BRH2G, BBG, //g
    Parameter p_bx2g = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_blh2g = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_brh2g = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bbg = model.add_parameters({hidden_dim});
    ps.push_back(p_bx2g);
    ps.push_back(p_blh2g);
    ps.push_back(p_brh2g);
    ps.push_back(p_bbg);

    //BX2O, BLH2O, BRH2O, BC2O, BBO,// o
    Parameter p_bx2o = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_blh2o = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_brh2o = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bc2o = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bbo = model.add_parameters({hidden_dim});
    ps.push_back(p_bx2o);
    ps.push_back(p_blh2o);
    ps.push_back(p_brh2o);
    ps.push_back(p_bc2o);
    ps.push_back(p_bbo);

    Parameter p_avg_l = model.add_parameters({layer_input_dim, layer_input_dim});
    Parameter p_avg_r = model.add_parameters({layer_input_dim, layer_input_dim});
    Parameter p_avg_lh = model.add_parameters({layer_input_dim, hidden_dim});
    Parameter p_avg_rh = model.add_parameters({layer_input_dim, hidden_dim});
    Parameter p_avg_lc = model.add_parameters({layer_input_dim, hidden_dim});
    Parameter p_avg_rc = model.add_parameters({layer_input_dim, hidden_dim});
    Parameter p_avg_b = model.add_parameters({layer_input_dim});
    ps.push_back(p_avg_l);
    ps.push_back(p_avg_r);
    ps.push_back(p_avg_lh);
    ps.push_back(p_avg_rh);
    ps.push_back(p_avg_lc);
    ps.push_back(p_avg_rc);
    ps.push_back(p_avg_b);

    Parameter p_label_start = model.add_parameters({config.LABEL_INPUT_DIM});

    Parameter p_bu2output = model.add_parameters({config.OUTPUT_HIDDEN_DIM, hidden_dim});
    Parameter p_bu_bias = model.add_parameters({config.OUTPUT_HIDDEN_DIM});

    Parameter p_output_h = model.add_parameters({config.SPAN_LABEL_SIZE, config.OUTPUT_HIDDEN_DIM});
    Parameter p_output_bias = model.add_parameters({config.SPAN_LABEL_SIZE});

    ps.push_back(p_label_start);
    ps.push_back(p_bu2output);
    ps.push_back(p_bu_bias);
    ps.push_back(p_output_h);
    ps.push_back(p_output_bias);

    assert(ps.size() == LB+1);
}


void BinarizedLexTreeLSTMBuilder::new_graph(ComputationGraph& cg)
{
    param_exprs.clear();
    vector<Expression>& vars = param_exprs;

    auto& p = params;
    for(size_t j = 0; j< p.size(); j++)//TODO: add output parameter here
    {
        Expression i_expr = parameter(cg, p[j]);
        vars.push_back(i_expr);
    }

    label_decoder.new_graph(cg);
}

Expression BinarizedLexTreeLSTMBuilder::build_graph(ComputationGraph& cg,
        ConTree* tree)
{
    //cerr<<"2" << endl;
    build_bottomup_graph(cg, *(tree->root));
    //cerr<<"3" << endl;
    return get_label_loss(cg, *(tree->root));
}


void BinarizedLexTreeLSTMBuilder::build_bottomup_graph(ComputationGraph& cg, struct SpanTreeNode& root)
{
    if(root.is_leaf)
    {
        const vector<Expression>& vars = param_exprs;
        Expression in = root.xi;
        //i
        Expression i_input_i = affine_transform({vars[BBI], vars[BX2I], in});
        Expression i_gate_i = logistic(i_input_i);

        //g
        Expression i_input_g = affine_transform({vars[BBG], vars[BX2G], in});
        Expression i_tanh_g = tanh(i_input_g);

        root.bu_ci = cmult(i_gate_i, i_tanh_g);
        //o
        Expression i_input_o = affine_transform({vars[BBO], vars[BX2O], in});
        Expression i_gate_o = logistic(i_input_o);

        root.bu_hi = cmult(i_gate_o, tanh(root.bu_ci));
    }
    else {
        for(auto& child: root.children)
        {
            build_bottomup_graph(cg, *child);
        }

        assert(root.children.size() == 2);
        const vector<Expression>& vars = param_exprs;
        SpanTreeNode* leftNode = root.children[0];
        SpanTreeNode* rightNode = root.children[1];

        Expression left_h = leftNode->bu_hi;
        Expression left_c = leftNode->bu_ci;
        Expression right_h = rightNode->bu_hi;
        Expression right_c = rightNode->bu_ci;
        //get root.xi
        Expression lex_gate = logistic(affine_transform({vars[AVG_B],
                                       vars[AVG_LH], left_h, vars[AVG_LC], left_c,
                                       vars[AVG_RH], right_h, vars[AVG_RC], right_c
                                                        }));

        root.xi = cmult(lex_gate, leftNode->xi) + cmult(1.0-lex_gate, rightNode->xi);

        Expression in = root.xi;
        if(!config.eval && config.ptreedrop > 0) //we don't do dropout for leaf nodes, because word_rep and pos_rep is already dropped for leaf nodes
            in = dropout(in, config.ptreedrop);

        Expression i_input_o;

        //i
        Expression i_input_i = affine_transform({vars[BBI],
                                                vars[BX2I], in,
                                                vars[BLH2I], left_h,
                                                vars[BLC2I], left_c,
                                                vars[BRH2I], right_h,
                                                vars[BRC2I], right_c
                                                });
        Expression i_gate_i = logistic(i_input_i);

        //l_f
        Expression i_input_lf = affine_transform({vars[BBLF],
                                vars[BX2LF], in,
                                vars[BLH2LF], left_h,
                                vars[BLC2LF], left_c,
                                vars[BRH2LF], right_h,
                                vars[BRC2LF], right_c
                                                 });
        Expression i_gate_lf = logistic(i_input_lf);

        //r_f
        Expression i_input_rf = affine_transform({vars[BBRF],
                                vars[BX2RF], in,
                                vars[BLH2RF], left_h,
                                vars[BLC2RF], left_c,
                                vars[BRH2RF], right_h,
                                vars[BRC2RF], right_c
                                                 });
        Expression i_gate_rf = logistic(i_input_rf);
        //g
        Expression i_input_g = affine_transform({vars[BBG],
                                                vars[BX2G], in,
                                                vars[BLH2G], left_h,
                                                vars[BRH2G], right_h //error here
                                                });
        Expression i_tanh_g = tanh(i_input_g);
        //6,
        root.bu_ci = cmult(i_gate_i, i_tanh_g) + cmult(i_gate_lf, left_c) + cmult(i_gate_rf, right_c);
        //7, o
        i_input_o = affine_transform({vars[BBO],
                                      vars[BX2O], in,
                                      vars[BLH2O],left_h,
                                      vars[BRH2O], right_h,
                                      vars[BC2O], root.bu_ci
                                     });

        Expression i_gate_o = logistic(i_input_o);
        //8,
        root.bu_hi = cmult(i_gate_o, tanh(root.bu_ci));
    }
}

Expression BinarizedLexTreeLSTMBuilder::get_label_loss(ComputationGraph& cg,
        struct SpanTreeNode& tree)
{
    auto& vars = param_exprs;
    vector<Expression> all_h;
    vector<unsigned> gold_labels;

    get_label_loss(cg, tree,  all_h, gold_labels);

    Expression batched_all_h = concatenate_to_batch(all_h);

    Expression i_r_t = rectify(affine_transform({vars[TB], vars[BU2T], batched_all_h}));

    Expression o_t = affine_transform({vars[LB], vars[T2L], i_r_t});

    Expression all_loss = pickneglogsoftmax(o_t, gold_labels);

    return sum_batches(all_loss);
}

void BinarizedLexTreeLSTMBuilder::get_label_loss(ComputationGraph& cg,
        struct SpanTreeNode& tree,
        vector<Expression>& all_h,
        vector<unsigned>& gold_labels)
{
    auto& vars = param_exprs;
    auto& phrase_nodes = tree.phrase_nodes;

    int phrase_index = 0;
    label_decoder.start_new_sequence();

    Expression decoder_input = concatenate({tree.bu_hi, param_exprs[LABEL_START]});
    if(!config.eval)
        decoder_input = dropout(decoder_input, config.pdrop);
    Expression label_h = label_decoder.add_input(decoder_input);
    //later, maybe we can use attention here, and include h[i], h[j] in the linear lstm

    while(phrase_index <= (int)phrase_nodes.size())
    {
        Expression loss; //generate labels here
        unsigned gold_label = ( phrase_index == (int) phrase_nodes.size()) ? LABEL_END : phrase_nodes[phrase_index]->span_label_id;

        all_h.push_back(label_h);
        gold_labels.push_back(gold_label);


        Expression label_input = lookup(cg, p_label_table, gold_label);
        decoder_input = concatenate({tree.bu_hi, label_input});
        if(!config.eval)
            decoder_input = dropout(decoder_input, config.pdrop);
        label_h = label_decoder.add_input(decoder_input);
        phrase_index++;
    }
    for(auto& child: tree.children)
        get_label_loss(cg, *child,  all_h, gold_labels);
}

Expression BinarizedLexTreeLSTMBuilder::generate_labels(ComputationGraph& cg, ConTree* pred_tree)
{
    assert(config.eval);
    //Expression loss_expr;
    //cerr<<"2" << endl;
    build_bottomup_graph(cg, *(pred_tree->root));
    //cerr<<"[tlog] 1" << endl;
    Expression loss_expr =  generate_labels(cg, *(pred_tree->root));
    //cerr<<"[tlog] 2" << endl;
    pred_tree->set_spans();
    //cerr<<"[tlog] 3" << endl;
    return loss_expr;
}


Expression BinarizedLexTreeLSTMBuilder::generate_labels(ComputationGraph& cg,
        struct SpanTreeNode& tree)
{
    vector<Expression> all_loss;
    generate_labels(cg, tree,  all_loss);
    /*if(config.verbose)
        cerr<< "all_loss.size " << all_loss.size() << endl;*/
    return sum(all_loss);
    //return input(cg, 0.0f);
}

void BinarizedLexTreeLSTMBuilder::generate_labels(ComputationGraph& cg,
        struct SpanTreeNode& tree,
        vector<Expression>& all_loss)
{
    if(config.verbose)
        cerr<<"[tlog] gen " << tree.span_left << ", " << tree.span_right << endl;

    auto& vars = param_exprs;
    auto& phrase_nodes = tree.phrase_nodes;


    unsigned phrase_index = 0;
    label_decoder.start_new_sequence();

    Expression label_h = label_decoder.add_input(concatenate({tree.bu_hi, param_exprs[LABEL_START]}));
    //later, maybe we can use attention here, and include h[i], h[j] in the linear lstm

    while(phrase_index <= config.MAX_LABLE_LENGTH)
    {
        Expression loss; //generate labels here

        Expression i_r_t;
        i_r_t = rectify(affine_transform({vars[TB], vars[BU2T], label_h}));

        Expression o_t = affine_transform({vars[LB], vars[T2L], i_r_t});

        cg.incremental_forward(o_t);
        auto label_values = as_vector(o_t.value());

        unsigned predicted_label = 0u;
        float max_value = label_values[0];
        for(unsigned label_index = 1u; label_index <  label_values.size(); label_index++)
        {
            //hard valid rules
            //1, the first label should not be "</s>"
            //2, the whole span should not be a label ended with star  label*
            //3, if the label size is greater than one, then the second label should be ended with star.  All unary rules should not be ended with stars
            //4, if the label is a star label  A*, then its parent label should be A.
            //5, if the sibling label is a star label, then the second label should not be a star label
            if(phrase_index == 0 && tree.span_left != tree.span_right && label_index == LABEL_END )
                continue;

            if(max_value < label_values[label_index])
            {
                max_value = label_values[label_index];
                predicted_label = label_index;
            }
        }

        loss = pickneglogsoftmax(o_t, predicted_label);

        all_loss.push_back(loss);

        if(config.verbose)
            cerr << "\t\tlabel: " << predicted_label << endl;
        assert(predicted_label < (int)config.SPAN_LABEL_SIZE);
        if(predicted_label == LABEL_END)
            break;

        struct PhraseNode* phrase_node = new PhraseNode;
        phrase_node->span_label_id = predicted_label;
        phrase_node->span_label_xi = lookup(cg, p_label_table, phrase_node->span_label_id);

        tree.phrase_nodes.push_back(phrase_node);
        /*if(config.verbose)
        {
            auto label_vec = as_vector(phrase_node->span_label_xi.value());
            copy(label_vec.begin(), label_vec.end(), ostream_iterator<float>(cerr, " "));
            cerr<<endl;
        }*/
        label_h = label_decoder.add_input(concatenate({tree.bu_hi, phrase_node->span_label_xi}));
        //label_h = label_decoder.add_input(concatenate({tree.bu_hi, param_exprs[LABEL_START]}));
        phrase_index++;

    }

    for(auto& child: tree.children)
        generate_labels(cg, *child, all_loss);
}

#endif //NEURAL_CKY_SPAN_BIN_LEX_TREE_LSTM_H
