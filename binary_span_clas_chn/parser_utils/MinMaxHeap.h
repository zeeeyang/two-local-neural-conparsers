#include <vector>
#include <algorithm>

template<class T>
class MinMaxHeap {
    std::vector<T> data;
    int max_size;

    bool isMaxLevel(int n) {}

    inline int getParent(int i) const {
        return (i - 1) >> 1;
    }
    inline int getGrandParent(int i)const {
        return (i - 3) >> 2;
    }
    inline bool hasParent(int i)const {
        return i - 1 >= 0;
    }
    inline bool hasGrandParent(int i)const {
        return i - 3 >= 0;
    }
    inline bool isMinLevel(int i)const {
        bool res = true;
        for (int j = 0; j < i; j = 2 * j + 2) {
            res = !res;
        }
        return res;
    }
    inline bool isChild(int i, int c) {
        return 2 * i + 1 <= c && c <= 2 * i + 2;
    }
    inline bool hasChild(int i) {
        return 2 * i + 1 < (int)data.size();
    }
    // trickleDown
    void trickleDownMin(int i) {
        for (int j = i, minIdx; hasChild(j); j = minIdx) {
            minIdx = 2 * j + 1;
            for (int g = 4 * j + 3, last = std::min(4 * j + 7, (int)data.size()); g < last; g++) {
                if (data[g] < data[minIdx])minIdx = g;
            }
            if (2 * j + 2 < (int)data.size() && data[2 * j + 2] <= data[minIdx])minIdx = 2 * j + 2;
            if (data[minIdx] >= data[j])break;
            std::swap(data[minIdx], data[j]);
            if (isChild(j, minIdx))break;
            if (data[minIdx]>data[getParent(minIdx)]) {
                std::swap(data[minIdx], data[getParent(minIdx)]);
            }
        }
    }
    void trickleDownMax(int i) {
        for (int j = i, maxIdx; hasChild(j); j = maxIdx) {
            maxIdx = 2 * j + 1;
            for (int g = 4 * j + 3, last = std::min(4 * j + 7, (int)data.size()); g < last; g++) {
                if (data[g] > data[maxIdx])maxIdx = g;
            }
            if (2 * j + 2 < (int)data.size() && data[2 * j + 2] >= data[maxIdx])maxIdx = 2 * j + 2;
            if (data[maxIdx] <= data[j])break;
            std::swap(data[maxIdx], data[j]);
            if (isChild(j, maxIdx))break;
            if (data[maxIdx]<data[getParent(maxIdx)]) {
                std::swap(data[maxIdx], data[getParent(maxIdx)]);
            }
        }
    }
    void trickleDown(int i) {
        if (isMinLevel(i)) {
            trickleDownMin(i);
        }
        else {
            trickleDownMax(i);
        }
    }
    // bubbleUp
    void bubbleUpMin(int i) {
        for (int j = i; hasGrandParent(j) && data[j] < data[getGrandParent(j)]; j = getGrandParent(j)) {
            std::swap(data[j], data[getGrandParent(j)]);
        }
    }
    void bubbleUpMax(int i) {
        for (int j = i; hasGrandParent(j) && data[j] > data[getGrandParent(j)]; j = getGrandParent(j)) {
            std::swap(data[j], data[getGrandParent(j)]);
        }
    }

    void bubbleUp(int i) {
        if (isMinLevel(i)) {
            if (hasParent(i) && data[i] > data[getParent(i)]) {
                std::swap(data[i], data[getParent(i)]);
                bubbleUpMax(getParent(i));
            }
            else {
                bubbleUpMin(i);
            }
        }
        else {
            if (hasParent(i) && data[i] < data[getParent(i)]) {
                std::swap(data[i], data[getParent(i)]);
                bubbleUpMin(getParent(i));
            }
            else {
                bubbleUpMax(i);
            }
        }
    }
public:
    MinMaxHeap(int size)
    {
        data.reserve(size);
        max_size = size;
    }
    int getSize() {
        return data.size();
    }

    bool isEmpty() {
        return data.empty();
    }

    void push(T x) {
        data.push_back(x);
        bubbleUp(data.size() - 1);
    }
    void deleteMin() {
        std::swap(data[0], data[(int)data.size() - 1]);
        data.pop_back();
        trickleDown(0);
    }
    void deleteMax() {
        if (data.size() == 1)return deleteMin();
        int maxIdx = 1;
        if ((int)data.size() >= 3 && data[2] > data[1])maxIdx = 2;
        std::swap(data[maxIdx], data[(int)data.size() - 1]);
        data.pop_back();
        trickleDown(maxIdx);
    }

    T getMin() const {
        return data[0];
    }

    T getMax() const {
        return data.size() >= 3 ? std::max<T>(data[1], data[2]) : data[data.size() == 2];
    }
    bool isFull() const {
        return (int)data.size() >= max_size;
    }
    const T& getElement(int index) const
    {
        assert(index < (int)data.size());
        return data[index];
    }
    T& getElement(int index)
    {
        assert(index < (int)data.size());
        return data[index];
    }
    inline bool isValid(float score) const {
        return data.size() > 0 && data[0] < score;
    }
};
