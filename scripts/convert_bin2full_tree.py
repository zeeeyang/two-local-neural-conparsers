#!/usr/bin/env python
import sys
import traceback


class TreeNode:
    def __init__(self):
        self.children=[]
        self.word=""
        self.isLeaf=False
        self.direction=""#l,r,s,l*,r*,t
        self.con_label=""# constituent label for non-terminal nodes, pos-tagging label for terminal nodes

class Tree:
    def __init__(self):
        self.root = TreeNode()

    def getLeaves(self):
        word_list = []
        self.__dfs_traverse(self.root, word_list)
        return word_list

    def printNodes(self, isDirectionIncluded=True):
        desc_list = []
        self.__dfs_print(self.root, desc_list, isDirectionIncluded)
        return desc_list
    
    def binary2full(self):
        self.__binary2full(self.root)

    def __binary2full(self, node):
        if node.isLeaf:
            return
        for child in node.children:
            self.__binary2full(child)
        leftNode = node.children[0]
        rightNode = None if len(node.children) == 1 else node.children[1]
        if "*" in leftNode.direction:
            node.children = []
            for child in leftNode.children:
                node.children.append(child)
            node.children.append(rightNode)
        elif rightNode is not None and "*" in rightNode.direction:
            node.children = []
            node.children.append(leftNode)
            for child in rightNode.children:
                node.children.append(child)

    def __dfs_traverse(self, node, word_list=[]):
        if node.isLeaf:
            #word_list.append(node.word+"/"+node.con_label)
            word_list.append(node.word)
        else:
            for child in node.children:
                self.__dfs_traverse(child, word_list)

    def __dfs_print(self, node, desc_list=[], isDirectionIncluded=True):
        if node.isLeaf:
            desc_list.append("(")
            desc_list.append(node.con_label)
            if isDirectionIncluded:
                desc_list.append(node.direction)
            desc_list.append(node.word)
            desc_list.append(")")
        else:
            desc_list.append("(")
            desc_list.append(node.con_label)
            if isDirectionIncluded:
                desc_list.append(node.direction)
            for child in node.children:
                self.__dfs_print(child, desc_list, isDirectionIncluded)
            desc_list.append(")")
    
    def accept(self, tree_line):
        self.__build_tree(tree_line, self.root)

    def __build_tree(self, line_desp, root):
        #(1 (2 (2 (2 Inco) (2 's)) (2 Net)) (1 (2 Soars) (1 (2 on) (1 (2 (2 (2 Higher) (2 (2 Metal) (2 Prices))) (2 ,)) (2 (2 Breakup) (2 Fee))))))
        line_desp = line_desp.strip()
        #print line_desp
        while line_desp[0] =='(' and  line_desp[-1] == ')':
            line_desp = line_desp[1:-1].strip()
            #print line_desp
        #find its own label
        label_pos = line_desp.find(" ")
        label = line_desp[0:label_pos].strip()
        line_desp = line_desp[label_pos+1:].strip()
        root.con_label = label
        #find its own direction
        dir_pos = line_desp.find(" ")
        dir = line_desp[0:dir_pos].strip()
        line_desp = line_desp[dir_pos+1:].strip()
        root.direction = dir
        
        #print label 
        #print root.label
        if line_desp[0] != "(": #leaf node
            root.word = line_desp.strip()
            root.isLeaf = True
            assert root.direction == "t"
            return
        root.isLeaf = False
        #traverse its children
        left_b = 0
        right_b = 0
        start_pos = 0
        end_pos = 0
        for i in range(len(line_desp)):
            if line_desp[i] == '(':
                left_b +=1
                if left_b == 1:
                    start_pos = i
            elif line_desp[i] ==')':
                right_b +=1
                if right_b == left_b:
                    end_pos = i
                    child_desp = line_desp[start_pos:end_pos+1].strip()
                    child_node = TreeNode()
                    #print start_pos, end_pos, child_desp
                    self.__build_tree(child_desp, child_node)
                    root.children.append(child_node)
                    left_b = 0
                    right_b = 0
                    start_pos = 0
                    end_pos = 0
            elif right_b > left_b:
                print line_desp[:i+1]
                print start_pos, i, line_desp[i], line_desp[i-1]
                print 'unbalanced tree', left_b, right_b, line_desp
    #print line_desp
    #
if __name__=='__main__':
    if len(sys.argv)!=3:
        print "%s input output" % sys.argv[0] 
    input_file = open(sys.argv[1], "r")
    output_file = open(sys.argv[2], "w")
    line_count = 0
    while True:
        try:
            line_count+=1
            #if line_count > 10:
            #    break
            line = input_file.next().strip()
            #print line
            if len(line) == 0:
                output_file.write("\n")
                continue
            tree = Tree()
            tree.accept(line)
            tree.binary2full()
            bracket_desc = tree.printNodes(False)
            bracket_str = " ".join(bracket_desc).replace("( ", "(").replace(" )", ")")
            output_file.write(bracket_str+"\n")
        except:
            #traceback.print_exc()
            break
