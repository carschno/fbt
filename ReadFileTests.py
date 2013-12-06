__author__ = 'carsten'

import unittest
import ReadFile


class MyTestCase(unittest.TestCase):
    def test_zipcsv(self):
        trainingfile1 = "/home/carsten/facebook/Train.zip"
        max_lines = 1000
        id1 = "1"
        title1 = "How to check if an uploaded file is an image without mime type?"
        tags1 = "php image-processing file-upload upload mime-types"

        trainingfile2 = "notexisting.zip"
        trainingfile3 = "/home/carsten/facebook/SampleSubmission.csv"

        entries = ReadFile.read_file(trainingfile1, zipped=True, max_lines=max_lines)
        self.assertEquals(max_lines, len(entries))
        self.assertEquals(id1, entries[0][0])
        self.assertEquals(title1, entries[0][1])
        self.assertEquals(tags1, entries[0][3])

        entries = ReadFile.read_file(trainingfile2, zipped=True)
        self.assertIsNone(entries)

        entries = ReadFile.read_file(trainingfile3, zipped=True)
        self.assertIsNone(entries)

    def test_read_csv(self):
        trainingfile1 = "/home/carsten/facebook/SampleSubmission.csv"
        max_lines = 1000
        #total_lines = 2013337
        trainingfile2 = "notexisting.csv"

        entries = ReadFile.read_file(trainingfile1, zipped=False, head=True, max_lines=max_lines)
        self.assertEquals(max_lines, len(entries))

        #entries = ReadFile.read_trainingfile(trainingfile1, zipped=False, head=True)
        #self.assertEquals(total_lines, len(entries))
        #
        #entries = ReadFile.read_trainingfile(trainingfile1, zipped=False, head=False)
        #self.assertLess(total_lines, len(entries))

        entries = ReadFile.read_file(trainingfile2, zipped=False, head=True, max_lines=max_lines)
        self.assertIsNone(entries)


if __name__ == '__main__':
    unittest.main()
