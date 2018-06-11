//
// Created by zeeeyang on 4/9/17.
//

#ifndef NEURAL_CKY_EXAMPLE_H
#define NEURAL_CKY_EXAMPLE_H

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <unordered_map>

using namespace std;

typedef int SPAN_KEY;

struct Example {

    vector<string> tokens;
    vector<unsigned> token_ids;

    vector<string> postags;
    vector<unsigned> postag_ids;


    vector<vector<string> > chars;
    vector<vector<unsigned> > char_ids;

    unordered_map<SPAN_KEY, vector<string> > span_labels;
    unordered_map<SPAN_KEY, vector<unsigned> > span_label_ids;

    unordered_map<SPAN_KEY, vector<unsigned> > predicted_span_label_ids;

    friend ostream& operator<<(ostream& cout, const Example& other)
    {
        copy(other.tokens.begin(), other.tokens.end(), ostream_iterator<string>(cout, " "));
        cout<<endl;
        copy(other.postags.begin(), other.postags.end(), ostream_iterator<string>(cout, " "));
        cout<<endl;
        for(auto& item: other.span_labels)
        {
            int span_left = item.first / other.size();
            int span_right = item.first % other.size();
            cout<<"\t["<<span_left<<", "<<span_right<<"] ";
            copy(item.second.begin(), item.second.end(), ostream_iterator<string>(cout, " "));
            cout<<endl;
        }
        return cout;
    }
    inline unsigned size() const {
        return tokens.size();
    }

    void make_padding(int pad_len)
    {
        //to do when necessary
    }
};
#endif //NEURAL_CKY_EXAMPLE_H
