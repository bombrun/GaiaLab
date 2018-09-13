import pandas as pd
import unittest
from . import filter_base as filter_base

"""
Unit testing for the filter data class.
"""

# Code is sparsely commented as functions are largely trivial and
# verbosely named.


class TestFilterInit(unittest.TestCase):
    def test_init_with_string_throws_error(self):
        with self.assertRaises(TypeError):
            filter_base.FilterData("string")

    def test_init_with_list_of_strings_throws_error(self):
        with self.assertRaises(TypeError):
            filter_base.FilterData(["list", "of", "strings"])

    def test_init_with_list_of_complex_numbers_throws_error(self):
        with self.assertRaises(TypeError):
            filter_base.FilterData([1 + 2j, 2 + 3j, 3+4j])

    def test_init_with_list_of_lists_throws_error(self):
        with self.assertRaises(TypeError):
            filter_base.FilterData([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    def test_init_with_single_number_throws_error(self):
        # There is an argument to be made that you should be able to
        # init an instance with only one number, but I disagree - how
        # can you remove noise from one data point?
        with self.assertRaises(TypeError):
            filter_base.FilterData(1)


class TestFilterComparisons(unittest.TestCase):
    def setUp(self):
        self.filter_list_1 = filter_base.FilterData([1, 2, 3, 4, 5, 6, 7])
        self.filter_tuple_1 = filter_base.FilterData((1, 2, 3, 4, 5, 6, 7))
        self.filter_list_different = filter_base.FilterData([2, 4, 6, 8, 10,
                                                             12, 14])
        self.filter_big_first_value = filter_base.FilterData([10000, 0, 0, 0,
                                                              0, 0, 0])

    def test_equality_of_same_data(self):
        filter_list_2 = self.filter_list_1.copy()
        self.assertEqual(filter_list_2, self.filter_list_1)

    def test_filter_init_from_list_and_filter_init_from_tuple_are_equal(self):
        self.assertEqual(self.filter_list_1, self.filter_tuple_1)

    def test_different_filter_base_are_not_equal(self):
        self.assertNotEqual(self.filter_list_1, self.filter_list_different)

    def test_greater_than_behaves_as_expected(self):
        # Greater than should use lexicographical ordering as is
        # standard in python.
        self.assertGreater(self.filter_big_first_value, self.filter_list_1)

    def test_greater_than_or_equal_to_behaves_as_expected(self):
        self.assertGreaterEqual(self.filter_big_first_value,
                                self.filter_list_1) and \
                self.assertGreaterEqual(self.filter_list_1,
                                        self.filter_list_1.copy())

    def test_less_than_behaves_as_expected(self):
        self.assertLess(self.filter_list_1, self.filter_big_first_value)

    def test_less_than_or_equal_to_behaves_as_expected(self):
        self.assertLessEqual(self.filter_list_1,
                             self.filter_big_first_value) and \
                 self.assertLessEqual(self.filter_list_1,
                                      self.filter_list_1.copy())

    def test_filter_and_string_are_not_equal(self):
        self.assertNotEqual(self.filter_list_1, "string")

    def test_filter_and_array_of_equivalent_data_are_not_equal(self):
        self.assertNotEqual(self.filter_list_1, self.filter_list_1._data)

    def test_filter_and_list_of_equivalent_data_are_not_equal(self):
        self.assertNotEqual(self.filter_list_1, [1, 2, 3, 4, 5, 6, 7])


class TestFilterArithmetic(unittest.TestCase):
    def setUp(self):
        self.filter_1 = filter_base.FilterData([1, 2, 3, 4, 5, 6, 7])
        self.equivalent_list = [1, 2, 3, 4, 5, 6, 7]
        self.double_filter = filter_base.FilterData([2, 4, 6, 8, 10, 12, 14])

    def test_add_filter_together(self):
        self.assertEqual(self.filter_1 + self.filter_1, self.double_filter)

    def test_add_list_to_filter(self):
        self.assertEqual(self.filter_1 + self.equivalent_list,
                         self.double_filter)

    def test_add_tuple_to_filter(self):
        self.assertEqual(self.filter_1 + tuple(self.equivalent_list),
                         self.double_filter)

    def test_add_constant_to_filter(self):
        filter_1_plus_2 = filter_base.FilterData([3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(self.filter_1 + 2, filter_1_plus_2)

    def test_add_string_to_filter_raises_error(self):
        with self.assertRaises(TypeError):
            self.filter_1 + "string"

    def test_add_list_of_different_length_to_filter_raises_error(self):
        with self.assertRaises(ValueError):
            self.filter_1 + [1, 2]

    def test_add_tuple_of_different_length_to_filter_raises_error(self):
        with self.assertRaises(ValueError):
            self.filter_1 + (1, 2)

    def test_add_filter_of_different_length_to_filter_raises_error(self):
        with self.assertRaises(ValueError):
            self.filter_1 + filter_base.FilterData((1, 2))

    def test_sub_filter_from_filter(self):
        self.assertEqual(self.double_filter - self.filter_1, self.filter_1)

    def test_sub_list_from_filter(self):
        self.assertEqual(self.double_filter - self.equivalent_list,
                         self.filter_1)

    def test_sub_tuple_from_filter(self):
            self.assertEqual(self.double_filter - tuple(self.equivalent_list),
                             self.filter_1)

    def test_sub_constant_from_filter(self):
        filter_1_minus_2 = filter_base.FilterData([-1, 0, 1, 2, 3, 4, 5])
        self.assertEqual(self.filter_1 - 2, filter_1_minus_2)

    def test_sub_string_from_filter_raises_error(self):
        with self.assertRaises(TypeError):
            self.filter_1 - "string"

    def test_sub_list_of_different_length_from_filter_raises_error(self):
        with self.assertRaises(ValueError):
            self.filter_1 - [1, 2]

    def test_sub_tuple_of_different_length_from_filter_raises_error(self):
        with self.assertRaises(ValueError):
            self.filter_1 - (1, 2)

    def test_sub_filter_of_different_length_from_filter_raises_error(self):
        with self.assertRaises(ValueError):
            self.filter_1 - filter_base.FilterData((1, 2))

    def test_mul_filter_by_constant(self):
        self.assertEqual(self.filter_1 * 2, self.double_filter)

    def test_left_mul_by_constant_equals_right_mul_by_constant(self):
        self.assertEqual(self.filter_1 * 2, 2 * self.filter_1)

    def test_mul_by_list_is_elementwise(self):
        self.assertEqual(self.filter_1 * self.equivalent_list,
                         filter_base.FilterData([1, 4, 9, 16, 25, 36, 49]))

    def test_mul_by_equivalent_list_is_equal_to_mul_by_filter(self):
        self.assertEqual(self.filter_1 * self.equivalent_list,
                         self.filter_1 * self.filter_1)

    def test_mul_by_equivalent_tuple_is_equal_to_mul_by_filter(self):
        self.assertEqual(self.filter_1 * tuple(self.equivalent_list),
                         self.filter_1 * self.filter_1)

    def test_mul_by_incorrect_length_tuple_raises_error(self):
        with self.assertRaises(TypeError):
            self.filter_1 * (1, 2)

    def test_mul_by_incorrect_type_raises_error(self):
        with self.assertRaises(TypeError):
            self.filter_1 * "string"

    def test_div_filter_by_constant(self):
        self.assertEqual(self.double_filter / 2, self.filter_1)

    def test_div_by_list_is_elementwise(self):
        self.assertEqual(filter_base.FilterData([1, 4, 9, 16, 25, 36, 49]) /
                         self.equivalent_list, self.filter_1)

    def test_div_by_equivalent_list_is_equal_to_div_by_filter(self):
        self.assertEqual(self.filter_1 / self.equivalent_list,
                         self.filter_1 / self.filter_1)

    def test_div_by_equivalent_tuple_is_equal_to_div_by_filter(self):
        self.assertEqual(self.filter_1 / tuple(self.equivalent_list),
                         self.filter_1 / self.filter_1)

    def test_div_by_incorrect_length_tuple_raises_error(self):
        with self.assertRaises(TypeError):
            self.filter_1 / (1, 2)

    def test_div_by_incorrect_type_raises_error(self):
        with self.assertRaises(TypeError):
            self.filter_1 / "string"


class TestFilterPandasInteraction(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
