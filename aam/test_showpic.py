import unittest

from showpic import *

class MathCalc(unittest.TestCase):
    def test_point_distance(self):
        pt1 = [1, 0]
        pt2 = [4, 4]
        expected = 5
        calculated = point_distance([pt1, pt2], 0, 1)
        self.assertEqual(expected, calculated)


    def test_point_to_line(self):
        pt1 = [1, 0]
        pt2 = [4, 4]
        pt3 = [-3, 3]
        expected = 0
        calculated = point_to_line_distance([pt1, pt2, pt3], 1, 0 , 0, 1)
        self.assertEqual(expected, calculated)

    def test_point_to_line_2(self):
        pt1 = [1, 0]
        pt2 = [4, 4]
        pt3 = [-3, 3]
        expected = 5
        calculated = point_to_line_distance([pt1, pt2, pt3], -3, 3 , 0, 1)
        self.assertEqual(expected, calculated)


    def test_point_n_to_line(self):
        pt1 = [1, 0]
        pt2 = [4, 4]
        pt3 = [-3, 3]
        expected = 0
        calculated = point_n_to_line_distance([pt1, pt2, pt3], 0, 0, 1)
        self.assertEqual(expected, calculated)

    def test_point_n_to_line_2(self):
        pt1 = [1, 0]
        pt2 = [4, 4]
        pt3 = [-3, 3]
        expected = 0
        calculated = point_n_to_line_distance([pt1, pt2, pt3], 1, 0, 1)
        self.assertAlmostEqual(expected, calculated)

    def test_point_n_to_line_3(self):
        pt1 = [1, 0]
        pt2 = [4, 4]
        pt3 = [-3, 3]
        expected = 5
        calculated = point_n_to_line_distance([pt1, pt2, pt3], 2, 0, 1)
        self.assertEqual(expected, calculated)


if __name__ == '__main__':
    unittest.main()
