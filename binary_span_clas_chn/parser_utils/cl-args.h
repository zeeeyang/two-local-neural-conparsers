#ifndef NEURAL_CKY_ARG_H
#define NEURAL_CKY_ARG_H
#pragma once

/**
 * \file cl-args.h
 * \brief This is a **very** minimal command line argument parser
 * Modified by Shuailong.
 * https://github.com/clab/dynet/blob/master/examples/cpp/utils/cl-args.h
 */
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>

using namespace std;

/**
 * \brief Structure holding any possible command line argument
 *
 */
struct Params {
    string exp_name = "neuralckyparsing";

    string train_file = "";
    string dev_file = "";
    string test_file = "";
    string dev_output_file = "";
    string test_output_file = "";
    string model_file = "";
    string basemodel_file = "";
    string embedding_file = "";
    string trainer = "adam";
    string idbuilder = "james";
    string unkmap_file = "";

    float pdrop = 0.5f;
    float ptreedrop = 0.0f;
    float P_UNK = 0.1f;
    float UNK_Z = 0.8375f;
    float RATE_DECAY = 0.5f;
    float RATE_THRESHOLD = 1e-5f;

    unsigned LAYERS = 1u;
    unsigned CHAR_LSTM_LAYERS = 1u;

    unsigned LABEL_LSTM_LAYERS = 1u;

    unsigned TOKEN_INPUT_DIM = 50u;
    unsigned HIDDEN_DIM = 150u;

    unsigned CHAR_INPUT_DIM = 20u;
    unsigned CHAR_HIDDEN_DIM = 25u;

    unsigned POSTAG_INPUT_DIM = 32u;

    unsigned OUTPUT_HIDDEN_DIM = 128u;
    unsigned LABEL_INPUT_DIM = 32u;

    unsigned BEAM_SIZE = 12u;
    float BETA = std::log(0.00025f);
    unsigned BATCH_SIZE = 20u;

    float LEARNING_RATE = 0.001f;
    bool  CLIP_ENABLED = true;
    float ETA_DECAY = 0.08f;
    // unsigned BATCH_SIZE = 1;
    // unsigned DEV_BATCH_SIZE = 16;
    int NUM_EPOCHS = 50;
    float SCHEDULED_SAMPLING_K = 15.0;// k = 3.0 is too small

    unsigned TOKEN_VOCAB_SIZE = 0u;
    unsigned CHAR_VOCAB_SIZE = 0u;
    unsigned SPAN_LABEL_SIZE = 0u;
    unsigned POSTAG_VOCAB_SIZE = 0u;

    unsigned MAX_LABLE_LENGTH = 4u;

    bool eval = false;
    bool verbose = false;
    bool is_test = false;
    bool inited_from_pretrained = false;
    bool use_char = false;

    friend ostream& operator<<(ostream& out, const struct Params& params);
};

ostream& operator<<(ostream& cerr, const struct Params& params)
{
    cerr << "Params:" << endl;
    cerr << "exp_name: " << params.exp_name << endl;
    cerr << "train_file: " << params.train_file << endl;
    cerr << "dev_file: " << params.dev_file << endl;
    cerr << "test_file: " << params.test_file << endl;
    cerr << "model_file: " << params.model_file << endl;
    cerr << "embedding_file: " << params.embedding_file << endl;
    cerr << "unk_file: " << params.embedding_file << endl;
    if(params.is_test) {
        cerr << "is_test: " << params.is_test << endl;
        cerr << "dev_output_file: " << params.dev_output_file << endl;
        cerr << "test_output_file: " << params.test_output_file << endl;
    }
    cerr << "pdrop: " << params.pdrop << endl;
    cerr << "ptreedrop: " << params.ptreedrop << endl;
    cerr << "P_UNK: " << params.P_UNK << endl;
    cerr << "LAYERS: " << params.LAYERS << endl;
    cerr << "TOKEN_INPUT_DIM: " << params.TOKEN_INPUT_DIM << endl;
    cerr << "HIDDEN_DIM: " << params.HIDDEN_DIM << endl;
    cerr << "CHAR_LSTM_LAYERS: " << params.CHAR_LSTM_LAYERS << endl;
    cerr << "CHAR_INPUT_DIM: " << params.CHAR_INPUT_DIM << endl;
    cerr << "CHAR_HIDDEN_DIM: " << params.CHAR_HIDDEN_DIM << endl;
    cerr << "OUTPUT_HIDDEN_DIM: " << params.OUTPUT_HIDDEN_DIM << endl;

    cerr << "NUM_EPOCHS: " << params.NUM_EPOCHS << endl;
    cerr << "SCHEDULED_SAMPLING_K: " << params.SCHEDULED_SAMPLING_K << endl;
    cerr << "BEAM_SIZE: " << params.BEAM_SIZE << endl;
    cerr << "BATCH_SIZE: " << params.BATCH_SIZE << endl;
    cerr << "BETA: " << params.BETA << endl;
    cerr << "RATE_DECAY: " << params.RATE_DECAY << endl;
    cerr << "RATE_THRESHOLD: " << params.RATE_THRESHOLD << endl;

    cerr << "Trainer: " << params.trainer << endl;
    cerr << "IdBuilder: " << params.idbuilder << endl;
    cerr << "LEARNING_RATE: " << params.LEARNING_RATE << endl;
    cerr<<  "CLIP_ENABLED: " << params.CLIP_ENABLED << endl;
    cerr<<  "ETA_DECAY: " << params.ETA_DECAY << endl;

    cerr << "inited_from_pretrained: " << params.inited_from_pretrained << endl;
    cerr << "use_char: " << params.use_char << endl;
    cerr << endl;
    return cerr;
}

/**
 * \brief Get parameters from command line arguments
 * \details Parses parameters from `argv` and check for required fields depending on the task
 *
 * \param argc Number of arguments
 * \param argv Arguments strings
 * \param params Params structure
 */
void get_args(int argc, char** argv, Params& params) {
    int i = 0;
    while (i < argc) {
        string arg = argv[i];
        if (arg == "--name" || arg == "-n") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.exp_name;
            i++;
        } else if (arg == "--train" || arg == "-t") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.train_file;
            i++;
        } else if (arg == "--dev" || arg == "-d") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.dev_file;
            i++;
        } else if (arg == "--test" || arg == "-ts") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.test_file;
            i++;
            //--unkmap_file ${datadir}/wsj.all.unk.dict
        } else if (arg == "--unkmap_file") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.unkmap_file;
            i++;
        } else if (arg == "--test_output_file") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.test_output_file;
            i++;
        } else if (arg == "--dev_output_file") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.dev_output_file;
            i++;
        } else if (arg == "--model" || arg == "-m") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.model_file;
            i++;
        }  else if (arg == "--basemodel" || arg == "-bm") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.basemodel_file;
            i++;
        } else if (arg == "--trainer") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.trainer;
            i++;
        } else if (arg == "--idbuilder") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.idbuilder;
            i++;
        } else if (arg == "--embedding" || arg == "-ef") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.embedding_file;
            i++;
        } else if (arg == "--dropout" || arg == "-dr") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.pdrop;
            i++;
        } else if (arg == "--tree_dropout") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.ptreedrop;
            i++;
        } else if (arg == "--punk") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.P_UNK;
            i++;
        } else if (arg == "--rate_decay") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.RATE_DECAY;
        } else if (arg == "--rate_threshold") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.RATE_THRESHOLD;
        } else if (arg == "--num_layers" || arg == "-l") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.LAYERS;
            i++;
        } else if (arg == "--char_lstm_layers") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.CHAR_LSTM_LAYERS;
            i++;
        } else if (arg == "--token_input_dim" || arg == "-tid") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.TOKEN_INPUT_DIM;
            i++;
        } else if (arg == "--hidden_dim" || arg == "-hd") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.HIDDEN_DIM;
            i++;
        }  else if (arg == "--output_hidden_dim" || arg == "-ohd") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.OUTPUT_HIDDEN_DIM;
            i++;
        }  else if (arg == "--char_input_dim" || arg == "-cid") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.CHAR_INPUT_DIM;
            i++;
        } else if (arg == "--char_hidden_dim" || arg == "-chd") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.CHAR_HIDDEN_DIM;
            i++;
        }  else if (arg == "--num_epochs" || arg == "-N") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.NUM_EPOCHS;
            i++;
        } else if (arg == "--k") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.SCHEDULED_SAMPLING_K;
            i++;
        } else if (arg == "--beam") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.BEAM_SIZE;
            i++;
        } else if (arg == "--batch") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.BATCH_SIZE;
            i++;
        } else if (arg == "--beta") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.BETA;
            params.BETA = std::log(params.BETA);
            i++;
        } else if (arg == "--istest") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.is_test;
            i++;
        } else if (arg == "--init_from_pretrained") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.inited_from_pretrained;
            i++;
        } else if (arg == "--use_char") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.use_char;
            i++;
        } else if (arg == "--clip_enabled") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.CLIP_ENABLED;
            i++;
        } else if (arg == "--learning_rate" || arg == "-lr") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.LEARNING_RATE;
            i++;
        } else if (arg == "--eta_decay") {
            if (i + 1 == argc) {
                std::cerr << "No matching argument for " << arg << std::endl;
                abort();
            }
            istringstream d(argv[i + 1]);
            d >> params.ETA_DECAY;
            i++;
        }
        i++;
    }

    if (params.train_file == "" || params.dev_file == "") {
        stringstream ss;
        ss << "Usage: " << argv[0] << " -t [train_file] -d [dev_file]";
        throw invalid_argument(ss.str());
    }

}

#endif //NEURAL_CKY_EVAL_H
