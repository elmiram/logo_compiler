"""
Tests for parser.py.

Only check that projects compile without failing.
"""

__author__ = "Elmira Mustakimova <cclikespizza@gmail.com>"

import os
import unittest
from unittest import TestCase


class TestProjects(TestCase):

    def test_projects(self):
        print()
        projects = ['test_projects/DUMMY/#1.mwx', 'test_projects/DUMMY/#2.mwx', 'test_projects/DUMMY/#3.mwx',
                    'test_projects/DUMMY/#4.mwx', 'test_projects/DUMMY/#5.mwx',
                    'test_projects/DUMMY/#6.mwx', 'test_projects/try2.mwx', 'test_projects/Project W.mwx',
                    'test_projects/button.mj3', 'test_projects/DropBalls.mwx']
        for project in projects:
            print(project)
            value = os.system("python parser.py \"" + project + '\" --libreoffice \"C:/Program Files (x86)/LibreOffice 5\" --log critical')
            if value != 0:
                self.fail('Compiler failed at project {}.'.format(project))


if __name__ == "__main__":
    unittest.main()