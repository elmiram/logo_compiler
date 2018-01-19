# -*- coding: utf-8 -*-

"""
Tests for LogoParser.py.
"""

__author__ = "Elmira Mustakimova <cclikespizza@gmail.com>"

import unittest
from unittest import TestCase
from LogoParser import Parser


class BaseUnitTest(TestCase):
    def setUp(self):
        print(self._testMethodName)
        self.PARSER = Parser()


class TestLogo(BaseUnitTest):

    def test_tokenize(self):
        self.assertEqual(['[', '11', ']'], self.PARSER.tokenize('11'))
        self.assertEqual(['[', 'покажи', '"язык Лого', ']'], self.PARSER.tokenize('покажи "|язык Лого|'))

        code = """to test tto "|t1 t1| if 2 = 2 [tto "t1] if 3 = 3 [if 2 = 2 [tto "t1]] end"""
        result = ['[', 'to', 'test', 'tto', '"t1 t1', 'if', '2', '=', '2', '[', 'tto', '"t1', ']',
                  'if', '3', '=', '3', '[', 'if', '2', '=', '2', '[', 'tto', '"t1', ']', ']', 'end', ']']

        self.assertEqual(result, self.PARSER.tokenize(code))


class TestPrimitives(BaseUnitTest):
    """test primitives"""

    def check(self, logo_code, js_code):
        # print(logo_code)
        self.assertEqual(js_code, self.PARSER.translate_from_anywhere(logo_code))
        # self.check('', '')

    def test_tto(self):
        """test tto"""
        self.check('tto "|name1 name2|',
                   'yield* prims.tto("name1 name2");')
        self.check("tto [Turtle1 Turtle2]",
                   'yield* prims.tto(["Turtle1", "Turtle2"]);')
        # self.check('name1,',
        #            'yield* prims.tto("name1");')

    def test_arithmetics_and_logics(self):
        self.check('1 + 1', 'prims.sum(1, 1)')
        self.check('sum 1 1', 'prims.sum(1, 1)')
        self.check('1 - 1', 'prims.difference(1, 1)')
        self.check('difference 1 1', 'prims.difference(1, 1)')
        self.check('1 * 1', 'prims.product(1, 1)')
        self.check('product 1 1', 'prims.product(1, 1)')
        self.check('quotient 1 1', 'prims.quotient(1, 1)')
        self.check('1 / 1', 'prims.quotient(1, 1)')
        self.check('less? 1 1', 'prims.less(1, 1)')
        self.check('1 < 1', 'prims.less(1, 1)')
        self.check('equal? 1 1', 'prims.equal(1, 1)')
        self.check('1 = 1', 'prims.equal(1, 1)')
        self.check('identical? 1 1', 'prims.identical(1, 1)')
        self.check('greater? 1 1', 'prims.greater(1, 1)')
        self.check('1 > 1', 'prims.greater(1, 1)')
        self.check('and 1 1', 'prims.and(1, 1)')
        self.check('or 1 1', 'prims.or(1, 1)')
        self.check('not 1', 'prims.not(1)')

    def test_precedence(self):
        self.check('1 + 3 * 8', 'prims.sum(1, prims.product(3, 8))')
        self.check('or 1 = 1 2 = 2', 'prims.or(prims.equal(1, 1), prims.equal(2, 2))')

    def test_parentheses(self):
        self.check('show (sum 1 2 3 4 5)',
                   'yield* prims.show(prims.sum(1, 2, 3, 4, 5));')
        self.check('1 + (3 * 8)', 'prims.sum(1, prims.product(3, 8))')
        self.check('1 - (3 + 8)', 'prims.difference(1, prims.sum(3, 8))')
        self.check('(1 - (3 + 8))', 'prims.difference(1, prims.sum(3, 8))')
        self.check('1 + (1 + 9) * 7', 'prims.sum(1, prims.product(prims.sum(1, 9), 7))')
        self.check('1 + (1 + 9) - 7', 'prims.difference(prims.sum(1, prims.sum(1, 9)), 7)')
        self.check('or 1 = 1 2 = 2', 'prims.or(prims.equal(1, 1), prims.equal(2, 2))')

    def test_sayas(self):
        self.check('sayas "word name',
                   'yield* prims.sayas("word", "name");')
        self.check('sayas ["word1 "word2] name',
                   'yield* prims.sayas(["word1", "word2"], "name");')

    def test_ask(self):
        self.check('ask [t1 t2 t3] [fd 50 rt 90 fd 50]',
                   'yield* prims.ask(["t1", "t2", "t3"], function*(){ yield* prims.fd(50); '
                   'yield* prims.rt(90); yield* prims.fd(50); });')

        self.check('ask "text2 [print "hello]',
                   'yield* prims.ask("text2", function*(){ yield* prims.print("hello"); });')

    def test_unknown_type(self):
        self.check('show ask "P1_Wario [1 + 1]',
                   'yield* prims.show(yield* prims.ask("P1_Wario", function*(){ prims.sum(1, 1) }));')

    def test_empty(self):
        self.check('ask [] []', 'yield* prims.ask("", function*(){  });')
        self.check('print []', 'yield* prims.print("");')

    def test_dolist_dotimes(self):
        self.check('dolist [i [a b c d]] [show :i]',
                   'yield* prims.dolist(scope, ["i", ["a", "b", "c", "d"]], '
                   'function*(){ yield* prims.show(scope.thing("i")); });')
        self.check('dotimes [i 8] [setc :i wait 5]',
                   'yield* prims.dotimes(scope, ["i", 8], function*(){ yield* '
                   'prims.setcolor(scope.thing("i")); yield* prims.wait(5); });')
        self.check('dotimes [i 10] [show :i]',
                   'yield* prims.dotimes(scope, ["i", 10], '
                   'function*(){ yield* prims.show(scope.thing("i")); });')

    def test_show(self):
        self.check('show "hello',
                   'yield* prims.show("hello");')
        self.check('show [hello there]',
                   'yield* prims.show(["hello", "there"]);')

    def test_word_or_list(self):
        pass
        # TODO:
        # 'TTO', 'ANNOUNCE', 'PICK', 'SAY', "PRINT", "INSERT", "QUESTION", "LIST?"

    def test_let(self):
        self.check('let [dist 100 head 90 delay 300]',
                   'scope.local("dist", 100); scope.local("head", 90); scope.local("delay", 300);')

    def test_if(self):
        self.check('if colorunder = 15 [bk 50]',
                   'yield* prims.if(prims.equal(prims.colorunder(), 15), function*(){ yield* prims.bk(50); });')
        self.check('ifelse colorunder = 15 [fd 50] [bk 50]',
                   'yield* prims.ifelse(prims.equal(prims.colorunder(), 15), '
                   'function*(){ yield* prims.fd(50); }, '
                   'function*(){ yield* prims.bk(50); });')

    def test_sentence(self):
        self.check('sentence [hi there] :name',
                   'prims.sentence(["hi", "there"], scope.thing("name"))')
        self.check('se [hi there] :name',
                   'prims.sentence(["hi", "there"], scope.thing("name"))')
        self.check('sentence [hi there] [hi there]',
                   'prims.sentence(["hi", "there"], ["hi", "there"])')

    def test_unknown_number_arguments(self):
        self.check('show (sum 1 2 3)',
                   'yield* prims.show(prims.sum(1, 2, 3));')
        self.check('show (product 1 2 3)',
                   'yield* prims.show(prims.product(1, 2, 3));')
        self.check('show (and 1 2 3)',
                   'yield* prims.show(prims.and(1, 2, 3));')
        self.check('show (or 1 2 3)',
                   'yield* prims.show(prims.or(1, 2, 3));')
        self.check('show (list 1 2 3)',
                   'yield* prims.show(prims.list(1, 2, 3));')
        self.check('show (word 1 2 3)',
                   'yield* prims.show(prims.word(1, 2, 3));')
        self.check('show (sentence 1 2 3)',
                   'yield* prims.show(prims.sentence(1, 2, 3));')


if __name__ == "__main__":
    unittest.main()