//
// Created by ji_ma on 5/9/17.
//

#ifndef NEURAL_CKY_ALL_DICT_H
#define NEURAL_CKY_ALL_DICT_H
#pragma once
unsigned int UTF8Len(unsigned char x) {
    if (x < 0x80)
        return 1;
    else if ((x >> 5) == 0x06)
        return 2;
    else if ((x >> 4) == 0x0e)
        return 3;
    else if ((x >> 3) == 0x1e)
        return 4;
    else if ((x >> 2) == 0x3e)
        return 5;
    else if ((x >> 1) == 0x7e)
        return 6;
    else
        return 0;
}
struct AtomDict
{
    unsigned kUNK; //id for unk item
    //unsigned kPAD; //id for padding
    dynet::Dict dict;
    bool set_unk_first; //some dict does not need a unk

    AtomDict(bool _set_unk_first = false): set_unk_first(_set_unk_first)
    {
        if(_set_unk_first) {
            kUNK = dict.convert("<unk>");
        }
    }

    void freeze()
    {
        dict.freeze();
    }
    void set_unk()
    {
        dict.set_unk("<unk>");
        if(!set_unk_first)
            kUNK = dict.convert("<unk>");
    }
    void set_padding()
    {
        //kPAD = dict.convert("**PAD**");
    }
};

struct DictSet {

    AtomDict postag_dict;
    AtomDict token_dict;
    AtomDict char_dict;
    AtomDict span_label_dict;

    DictSet(): token_dict(true), char_dict(true)
    {

    }

    void freeze()
    {
        postag_dict.freeze();
        token_dict.freeze();
        char_dict.freeze();
        span_label_dict.freeze();
    }

    void set_unk()
    {
        postag_dict.set_unk();
        token_dict.set_unk();
        char_dict.set_unk();
        span_label_dict.set_unk();
    }

    void set_padding()
    {
        postag_dict.set_padding();
        token_dict.set_padding();
        char_dict.set_padding();
        span_label_dict.set_padding();
    }
};

#endif //NEURAL_CKY_ALL_DICT_H
