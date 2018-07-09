import pandas as pd
import unittest
import hits.noiseremoval.kalman as k
"""
Unit testing for the Kalman data class.
"""
#Code is sparsely commented as functions are thus far trivial and named verbosely.

#TODO implement unit testing for more complex functions of Kalman data class.

class TestKalmanInit(unittest.TestCase):
    def test_init_with_string_throws_error(self):
        with self.assertRaises(TypeError):
            k.KalmanData("string")

    def test_init_with_list_of_strings_throws_error(self):
        with self.assertRaises(TypeError):
            k.KalmanData(["list", "of", "strings"])

    def test_init_with_list_of_complex_numbers_throws_error(self):
        with self.assertRaises(TypeError):
            k.KalmanData([1 + 2j, 2 + 3j, 3+4j])

    def test_init_with_list_of_lists_throws_error(self):
        with self.assertRaises(TypeError):
            k.KalmanData([[1,2,3],[1,2,3],[1,2,3]])

    def test_init_with_single_number_throws_error(self): # there is an argument to be made that you should be able to init an instance
                                                         # with only one number, but I disagree - how can you remove noise from one data point?
        with self.assertRaises(TypeError):
            k.KalmanData(1)

class TestKalmanComparisons(unittest.TestCase):
    def setUp(self):
        self.kalman_list_1 = k.KalmanData([1,2,3,4,5,6,7])
        self.kalman_tuple_1 = k.KalmanData((1,2,3,4,5,6,7))  
        self.kalman_list_different = k.KalmanData([2,4,6,8,10,12,14])
        self.kalman_big_first_value = k.KalmanData([10000,0,0,0,0,0,0])

    def test_equality_of_same_data(self):
        kalman_list_2 = self.kalman_list_1.copy()
        self.assertEqual(kalman_list_2, self.kalman_list_1)

    def test_kalman_init_from_list_and_kalman_init_from_tuple_are_equal(self):
        self.assertEqual(self.kalman_list_1, self.kalman_tuple_1)

    def test_different_kalmans_are_not_equal(self):
        self.assertNotEqual(self.kalman_list_1, self.kalman_list_different)

    def test_greater_than_behaves_as_expected(self): # greater than should use lexicographical ordering as is standard in python
        self.assertGreater(self.kalman_big_first_value, self.kalman_list_1)

    def test_greater_than_or_equal_to_behaves_as_expected(self):
        self.assertGreaterEqual(self.kalman_big_first_value, self.kalman_list_1) and self.assertGreaterEqual(self.kalman_list_1, self.kalman_list_1.copy())

    def test_less_than_behaves_as_expected(self):
        self.assertLess(self.kalman_list_1, self.kalman_big_first_value)

    def test_less_than_or_equal_to_behaves_as_expected(self):
        self.assertLessEqual(self.kalman_list_1, self.kalman_big_first_value) and self.assertLessEqual(self.kalman_list_1, self.kalman_list_1.copy())

    def test_kalman_and_string_are_not_equal(self):
        self.assertNotEqual(self.kalman_list_1, "string")

    def test_kalman_and_array_of_equivalent_data_are_not_equal(self):
        self.assertNotEqual(self.kalman_list_1, self.kalman_list_1._data)
    
    def test_kalman_and_list_of_equivalent_data_are_not_equal(self):
        self.assertNotEqual(self.kalman_list_1, [1,2,3,4,5,6,7])

class TestKalmanArithmetic(unittest.TestCase):
    def setUp(self):
        self.kalman_1 = k.KalmanData([1,2,3,4,5,6,7])
        self.equivalent_list = [1,2,3,4,5,6,7]
        self.double_kalman = k.KalmanData([2,4,6,8,10,12,14])

    def test_add_kalman_together(self):
        self.assertEqual(self.kalman_1 + self.kalman_1, self.double_kalman)
       
    def test_add_list_to_kalman(self):
        self.assertEqual(self.kalman_1 + self.equivalent_list, self.double_kalman)
    
    def test_add_tuple_to_kalman(self):
        self.assertEqual(self.kalman_1 + tuple(self.equivalent_list), self.double_kalman)

    def test_add_constant_to_kalman(self):
        kalman_1_plus_2 = k.KalmanData([3,4,5,6,7,8,9])
        self.assertEqual(self.kalman_1 + 2, kalman_1_plus_2)

    def test_add_string_to_kalman_raises_error(self):
        with self.assertRaises(TypeError):
            self.kalman_1 + "string"

    def test_add_list_of_different_length_to_kalman_raises_error(self):
        with self.assertRaises(ValueError):
            self.kalman_1 + [1,2]

    def test_add_tuple_of_different_length_to_kalman_raises_error(self):
        with self.assertRaises(ValueError):
            self.kalman_1 + (1,2)

    def test_add_kalman_of_different_length_to_kalman_raises_error(self):
        with self.assertRaises(ValueError):
            self.kalman_1 + k.KalmanData((1,2))

    def test_sub_kalman_from_kalman(self):
        self.assertEqual(self.double_kalman - self.kalman_1, self.kalman_1)
       
    def test_sub_list_from_kalman(self):
        self.assertEqual(self.double_kalman - self.equivalent_list, self.kalman_1)
    
    def test_sub_tuple_from_kalman(self):
            self.assertEqual(self.double_kalman - tuple(self.equivalent_list), self.kalman_1)

    def test_sub_constant_from_kalman(self):
        kalman_1_minus_2 = k.KalmanData([-1,0,1,2,3,4,5])
        self.assertEqual(self.kalman_1 - 2, kalman_1_minus_2)

    def test_sub_string_from_kalman_raises_error(self):
        with self.assertRaises(TypeError):
            self.kalman_1 - "string"

    def test_sub_list_of_different_length_from_kalman_raises_error(self):
        with self.assertRaises(ValueError):
            self.kalman_1 - [1,2]

    def test_sub_tuple_of_different_length_from_kalman_raises_error(self):
        with self.assertRaises(ValueError):
            self.kalman_1 - (1,2)

    def test_sub_kalman_of_different_length_from_kalman_raises_error(self):
        with self.assertRaises(ValueError):
            self.kalman_1 - k.KalmanData((1,2))

    def test_mul_kalman_by_constant(self):
        self.assertEqual(self.kalman_1 * 2, self.double_kalman)

    def test_left_mul_by_constant_equals_right_mul_by_constant(self):
        self.assertEqual(self.kalman_1 * 2, 2 * self.kalman_1)

    def test_mul_by_list_is_elementwise(self):
        self.assertEqual(self.kalman_1 * self.equivalent_list, k.KalmanData([1,4,9,16,25,36,49]))

    def test_mul_by_equivalent_list_is_equal_to_mul_by_kalman(self):
        self.assertEqual(self.kalman_1 * self.equivalent_list, self.kalman_1 * self.kalman_1)

    def test_mul_by_equivalent_tuple_is_equal_to_mul_by_kalman(self):
        self.assertEqual(self.kalman_1 * tuple(self.equivalent_list), self.kalman_1 * self.kalman_1)
    
    def test_mul_by_incorrect_length_tuple_raises_error(self):
        with self.assertRaises(TypeError):
            self.kalman_1 * (1,2)

    def test_mul_by_incorrect_type_raises_error(self):
        with self.assertRaises(TypeError):
            self.kalman_1 * "string"
            
    def test_div_kalman_by_constant(self):
        self.assertEqual(self.double_kalman / 2, self.kalman_1)

    def test_div_by_list_is_elementwise(self):
        self.assertEqual(k.KalmanData([1,4,9,16,25,36,49])/ self.equivalent_list, self.kalman_1)

    def test_div_by_equivalent_list_is_equal_to_div_by_kalman(self):
        self.assertEqual(self.kalman_1 / self.equivalent_list, self.kalman_1 / self.kalman_1)

    def test_div_by_equivalent_tuple_is_equal_to_div_by_kalman(self):
        self.assertEqual(self.kalman_1 / tuple(self.equivalent_list), self.kalman_1 / self.kalman_1)
    
    def test_div_by_incorrect_length_tuple_raises_error(self):
        with self.assertRaises(TypeError):
            self.kalman_1 / (1,2)

    def test_div_by_incorrect_type_raises_error(self):
        with self.assertRaises(TypeError):
            self.kalman_1 / "string"

class TestKalmanPandasInteraction(unittest.TestCase):
    def test_read_pandas_raises_error_if_columns_are_wrong(self):
        bad_data = pd.DataFrame(data=dict(bad=[1,2,3],
                                          column = [1,2,3],
                                          titles = [1,2,3]))
        with self.assertRaises(ValueError):
            k.KalmanData(bad_data)
if __name__ == "__main__":
    unittest.main()
