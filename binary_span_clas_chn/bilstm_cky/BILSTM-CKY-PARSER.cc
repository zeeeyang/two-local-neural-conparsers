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
#include "../parser_utils/corpus.h"
#include "../parser_utils/span_parser.h"
#include "../parser_utils/id_builder.h"

using namespace dynet;
using namespace dynet::expr;
using namespace std;

//==global variable definition
struct Params params;


IdBuilder* id_builder;

//==end global variable definition

void Clear(vector<ConTree*>& trees)
{
    for(auto& tree: trees)
        delete tree;
    trees.clear();
}


Trainer* create_trainer(Model& model)
{
    Trainer* trainer = NULL;
    if(params.trainer == "adam") {
        trainer = new AdamTrainer(model, params.LEARNING_RATE, 0.9, 0.999, 1e-8);
    } else if(params.trainer == "adadelta") {
        trainer = new AdadeltaTrainer(model, 1e-7, 0.99);
    } else if(params.trainer == "sgd") {
        trainer = new SimpleSGDTrainer(model, params.LEARNING_RATE);
        if(params.ETA_DECAY>0) trainer->eta_decay = params.ETA_DECAY;
    } else if(params.trainer == "sgdmem") {
        trainer = new MomentumSGDTrainer(model, params.LEARNING_RATE);
        if(params.ETA_DECAY>0) trainer->eta_decay = params.ETA_DECAY;
    }
    trainer->sparse_updates_enabled = false;
    trainer->clipping_enabled = params.CLIP_ENABLED;
    //trainer->clip_threshold *= params.BATCH_SIZE;
    return trainer;
}

template<class T>
float evaluate(T& span_parser, vector<ConTree*>& dev_examples,  const string& dev_output_file,
               const string& command_line, float epoch = -1)
{
    ofstream fout(dev_output_file);
    FMeasure dev_measure("labeled span");
    FMeasure dev_unlabeled_measure("unlabeled span");
    FMeasure dev_nontree_unlabeled_measure("nontree unlabeled span");
    float dloss = 0;
    params.eval = true;

    for(unsigned sent_index = 0; sent_index < dev_examples.size(); ++sent_index) {
        if(sent_index % 200 == 0) {
            cerr << "[log] " << sent_index << endl;
	    if(epoch == -1)
            	params.verbose = true;
        } else {
            params.verbose = false;
        }
        ComputationGraph cg;
        span_parser.new_graph(cg);

        bool valid = false;
        vector<ConTree*> pred_trees;
        Expression yloss = span_parser.build_graph(cg, dev_examples[sent_index], dev_measure, dev_unlabeled_measure, dev_nontree_unlabeled_measure, valid, pred_trees);

        //Expression yloss = span_parser.build_topdown_decoding_graph(cg, dev_examples[sent_index], dev_measure, valid);
        if(valid)
        {
            cg.incremental_forward(yloss);
            float current_loss = as_scalar(yloss.value());
            dloss += current_loss;
        }
        if(pred_trees.size() >0 && pred_trees[0]) {
            pred_trees[0]->leaf_input_nodes = dev_examples[sent_index]->leaf_input_nodes;
            vector<string> strs;
            pred_trees[0]->str(id_builder->get_all_dict(), strs);
            copy(strs.begin(), strs.end(), ostream_iterator<string>(fout, ""));
            fout<<endl;
            pred_trees[0]->leaf_input_nodes.clear();
            delete pred_trees[0];
        }
        pred_trees.clear();
    }
    fout.flush();
    fout.close();

    if(epoch == -1)
        cerr << "\n***Test ";
    else
        cerr << "\n***DEV [epoch=" << epoch << "] ";
    cerr<<"E = " << dloss ;
    cerr<<endl;
    dev_measure.report();
    dev_unlabeled_measure.report();
    dev_nontree_unlabeled_measure.report();
    params.eval = false;
    params.verbose = false;
    //float f1 = exec((command_line +" "+ dev_output_file).c_str());
    float f1 = dev_measure.f();
    cerr<< "rF1: " << f1 << endl;
    return f1;
}

int main(int argc, char** argv)
{
    dynet::initialize(argc, argv);

    for(int i = 0; i< argc; i++)
        cerr<<"Command: " << argv[i] << endl;

    get_args(argc, argv, params);
    cerr<< params << endl;

    id_builder = create_id_builder(params);

    ReadUnkMap(params.unkmap_file, id_builder);
    ReadEmbeddings(params, id_builder);


    vector<ConTree*> training_examples, dev_examples, test_examples;


    ReadCorpus(params.train_file,  training_examples);
    ReadCorpus(params.dev_file, dev_examples);
    ReadCorpus(params.test_file, test_examples);

    id_builder->build_id_for_examples(training_examples, dev_examples, test_examples);

    auto& all_dict = id_builder->get_all_dict();
    params.TOKEN_VOCAB_SIZE = all_dict.token_dict.dict.size();
    params.CHAR_VOCAB_SIZE = all_dict.char_dict.dict.size();
    params.POSTAG_VOCAB_SIZE = all_dict.postag_dict.dict.size();
    params.SPAN_LABEL_SIZE = all_dict.span_label_dict.dict.size();


    cerr<<"training tree size: "<< training_examples.size() << endl;
    cerr<<"dev tree size: "<< dev_examples.size() << endl;
    cerr<<"test tree size: "<< test_examples.size() << endl;

    cerr<<"word vocab_size: "<< params.TOKEN_VOCAB_SIZE << endl;
    cerr<<"char vocab_size: "<< params.CHAR_VOCAB_SIZE << endl;
    cerr<<"postag vocab_size: "<< params.POSTAG_VOCAB_SIZE << endl;
    cerr<<"span_label_size: "<< params.SPAN_LABEL_SIZE << endl;

    cerr<<"[span labels]: " << endl;
    for(auto i = 0u; i < params.SPAN_LABEL_SIZE; i++)
        cerr <<"\t" << all_dict.span_label_dict.dict.convert(i) << endl;
    //exit(0);
    ostringstream os;
    string fname;
    if (params.model_file != "") {
        fname = params.model_file;
    }
    else {
        ostringstream os;
        os << "neural.cky.parser."
           << '_' << params.TOKEN_INPUT_DIM
           << '_' << params.HIDDEN_DIM
           << '_' << params.LAYERS
           << "-pid" << getpid()
           << ".params";
        fname = "models/"+os.str();
    }
    cerr << "pid: " << getpid() << endl;
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;

    dynet::real learning_rate = params.LEARNING_RATE;
    dynet::real learning_scale = 1.0;

    Trainer* trainer = create_trainer(model);

    ConstituentSpanParser<LSTMBuilder> span_parser(model, params, all_dict, id_builder->get_pretrained_embeddings());

    bool isTest = params.is_test;
    if(isTest) {
        ifstream in(fname);
        boost::archive::text_iarchive ia(in);
        ia >> model;

        evaluate(span_parser, dev_examples, params.dev_output_file, "./eval_ch_dev_file.sh",  -1);

        evaluate(span_parser, test_examples, params.test_output_file, "./eval_ch_test_file.sh",  -1);

        return 0;
    }


    unsigned report_every_i = min(1000, int(training_examples.size()));
    unsigned dev_report_every_i = 10;
    int report = 0;
    unsigned si = training_examples.size();
    vector<unsigned> order(training_examples.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    unsigned lines = 0;
    int count = 0;
    FMeasure training_measure("labeled span");
    FMeasure training_unlabeled_measure("unlabeled span");
    FMeasure training_nontree_unlabeled_measure("nontree unlabeled span");
    float best  = 0.f;
    int best_keep = 0;
    while(count <= params.NUM_EPOCHS ) {
        Timer iteration("completed in");
        double loss = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if(si%1000 == 0) cerr<<"si: " << si << endl;
            if (si == training_examples.size()) {
                count++;
                si = 0;
                if (first) {
                    first = false;
                }
                else {
                    training_measure.report();
                    training_unlabeled_measure.report();
                    params.eval = false;
                    params.verbose = false;
                    trainer->update_epoch();
                }
                training_measure.reset();
                training_unlabeled_measure.reset();
                training_nontree_unlabeled_measure.reset();
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }

            ComputationGraph cg;
            span_parser.new_graph(cg);
            auto& tree = training_examples[order[si]];

            id_builder->stochastic_replace(tree, params);

            bool valid = false;
            vector<ConTree*> pred_trees;
            Expression yloss = span_parser.build_graph(cg, tree, training_measure, training_unlabeled_measure, training_nontree_unlabeled_measure, valid, pred_trees);
            if(valid) {
                cg.incremental_forward(yloss);
                loss += as_scalar(yloss.value());
                cg.backward(yloss);
                trainer->update(1.0);
            }
            if(pred_trees.size() > 0 && pred_trees[0])
                delete pred_trees[0];
            pred_trees.clear();
            ++si;
            ++lines;
        }
        trainer->status();
        cerr << " E = " << loss  <<" " << endl;

        // show score on dev data?
        ++report;
        if(report % dev_report_every_i  == 0 )
        {
            float f = evaluate(span_parser,  dev_examples, params.dev_output_file, "./eval_ch_dev_file.sh", lines / (float)training_examples.size());

            if ( best <= f ) {
                cerr<<"[log] Exceed "<< best << "\t" << f << endl;
                best = f;
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
                best_keep = 0;
            }
            else if(count >=20) {
                best_keep ++;
            }

            if(best_keep >= 20) //early stopping
                break;
        }
    }

    delete trainer;
    delete id_builder;
    Clear(training_examples);

    Clear(dev_examples);

    Clear(test_examples);

    return 0;
}
