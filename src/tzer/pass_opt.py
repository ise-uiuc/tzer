
import hashlib
from typing import List
import tvm

from collections import OrderedDict

from tvm import relay


def md5(data):
    return hashlib.md5(data).hexdigest()


class OptimizeNode:
    def __init__(self, opt_pass) -> None:
        self.opt_pass = opt_pass
        self.prefix = None
        self.suffixies = []
        self.cache = None
        self.is_last_one = False

    def append_suffix(self, suffix_pass):
        suffix_node = self.search_suffix(suffix_pass)
        if suffix_node != None:
            return suffix_node

        suffix_node = OptimizeNode(suffix_pass)
        suffix_node.prefix = self
        self.suffixies.append(suffix_node)
        return suffix_node

    def search_suffix(self, suffix_pass):
        for node in self.suffixies:
            if node.opt_pass.__name__ == suffix_pass.__name__:
                return node
        return None

class OptimizeTree:
    def __init__(self, module, target) -> None:
        self.root = OptimizeNode(None)
        self.root.cache = module
        self.target = target

    # add sequence of pass
    def add_sequence(self, sequence):
        node = self.root
        for item in sequence:
            node = node.append_suffix(item)
        node.is_last_one = True
        return node

    def add_sequences(self, sequences):
        return [self.add_sequence(seq) for seq in sequences]

    # optimize
    def optimize(self, node=None):
        if node == None:
            node = self.root
        elif node.cache == None:
            with tvm.transform.PassContext(opt_level=4):
                with self.target:
                    node.cache = tvm.transform.Sequential(
                        passes=[node.opt_pass()],
                        opt_level=4
                    )(node.prefix.cache)
        self.reporter.record_coverage()
        for suffix in node.suffixies:
            self.optimize(suffix)

    
    def search_sequence(self, sequence):
        node = self.root
        for item in sequence:
            node = node.search_suffix(item)
        assert(node.is_last_one == True)
        return node


    def search_sequence_backtrace(self, sequence):
        leaf_node = self.search_sequence(sequence)
        return leaf_node.prefix

    # export all cache
    def export_cache(self):
        return self.export_cache_node(self.root)        

    def export_cache_node(self, node):
        cache_modules = []
        if node.is_last_one:
            cache_modules.append(node.cache)

        for suffix in node.suffixies:
            cache_modules += self.export_cache_node(suffix)

        return cache_modules


    def print_node(self, node):
        print(node.opt_pass)
        print(md5(node.cache.astext().encode()))
        for s in node.suffixies:
            self.print_node(s)


class PassOptimizer:
    def __init__(self, capacity=20) -> None:
        self.capacity = capacity
        self.trees = OrderedDict()


    def get(self,key):
        if key in self.trees:
            value = self.trees.pop(key)
            self.trees[key] = value
        else:
            value = None
         
        return value
    
    def set(self, key, value):
        if key in self.trees:
            value = self.trees.pop(key)
            self.trees[key] = value
        else:
            if len(self.trees) == self.capacity:
                self.trees.popitem(last = False)
                self.trees[key] = value
            else:
                self.trees[key] = value


    def optimize(self, module: tvm.ir.module.IRModule, sequences: List[List[relay.transform.FunctionPass]], target: tvm.target.Target, reporter) -> tvm.ir.module.IRModule:
        if self.capacity > 0:
            key = md5(module.astext().encode())
            tree = self.get(key)
            if tree == None:
                tree = OptimizeTree(module, target)
                self.set(key, tree)
        else:
            tree = OptimizeTree(module, target)

        tree.reporter = reporter
        
        leaf_nodes = tree.add_sequences(sequences)

        tree.optimize()
        
        return [leaf_node.cache for leaf_node in leaf_nodes]

    def backtrace_optimize(self, module: tvm.ir.module.IRModule, sequences: List[List[relay.transform.FunctionPass]], target: tvm.target.Target) -> tvm.ir.module.IRModule:
        tree = OptimizeTree(module, target)
        leaf_nodes = tree.add_sequences(sequences)

        set_list = [leaf_nodes]
        current_set = set_list[-1]

        while len(current_set) > 1:
            new_set = set()
            for node in current_set:
                if node.prefix != None:
                    new_set.add(node.prefix)

            current_set = list(new_set)
            set_list.append(current_set)

        for current_set in set_list[::-1]:
            for seq in current_set:
                if seq.cache == None:
                    with tvm.transform.PassContext(opt_level=4):
                        with target:
                            seq.cache = tvm.transform.Sequential(
                                passes=[seq.opt_pass()],
                                opt_level=4
                            )(seq.prefix.cache)


        return [leaf_node.cache for leaf_node in leaf_nodes]

