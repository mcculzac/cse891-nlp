"""
Zachary McCullough
mccul157@msu.edu
2020-01-13
test cases
"""

#########
# imports
#########

import unittest
from main import *


#########
# classes
#########

class MyTestCase(unittest.TestCase):
    def test_word_freq(self):
        case = "this is a test\nof the word_freq case. this\tthis thisthis is a"
        self.assertEqual({"this": 3,
            "is": 2,
            "a": 2,
            "test": 1,
            "of": 1,
            "the": 1,
            "word_freq": 1,
            "case": 1,
            "thisthis": 1
        }, word_freq(case))

    def test_all_word_freq(self):
        case = ["this is", "a test", "\nof the word_freq", "case.", "this\tthis thisthis", "is a"]
        self.assertEqual({"this": 3,
            "is": 2,
            "a": 2,
            "test": 1,
            "of": 1,
            "the": 1,
            "word_freq": 1,
            "case": 1,
            "thisthis": 1
        }, all_word_freq(case, None))

    def test_vector_form(self):
        case = {"z": 4, "a": 0, "c": 2, "b": 1, "u": 3}
        # have to save and use tolist because otherwise get error from numpy
        result_ints, result_keys = vector_form(case)
        self.assertEqual(([0, 1, 2, 3, 4], ["a", "b", "c", "u", "z"]), (result_ints.tolist(), result_keys))

    def test_tf_idf(self):
        case = ([[1, 1, 0], [0, 1, 0]], ["0", "1"])
        self.assertEqual([[math.log(1), math.log(2/3.0), 0], [0, math.log(2/3.0), 0]], tf_idf(case[0], case[1]))

    def test_get_universal_word_bag(self):
        case_arg1, case_arg2 = ['a', 'b', 'c'], ['b', 'c', 'd', 'e']
        self.assertEqual(['a', 'b', 'c', 'd', 'e'], get_universal_word_bag(case_arg1, case_arg2))

    def get_top_n_results(self):
        case1 = {0: {'a': 1, 'b': 2}, 1: {'a': 2, 'b': 0, 'c':1, 'd': 1}}
        self.assertEqual({'a': 2, 'b': 2, 'c':1}, get_top_n_words(case1, 3))



#####
# run
#####

if __name__ == '__main__':
    unittest.main()
