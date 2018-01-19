# -*- coding: utf-8 -*-

"""
Parser and translator for Logo programming language.
"""

__author__ = "Elmira Mustakimova <cclikespizza@gmail.com>"

import logging
import re
import os

from enum import Enum
from collections import deque
from collections import defaultdict

try:
    from jsbeautifier import beautify
except ImportError:
    try:
        import pip
        pip.main(['install', "jsbeautifier"])
    except:
        pass

    try:
        from jsbeautifier import beautify
    except ImportError:
        def beautify(s):
            return s

logger = logging.getLogger("LogoParser")
logger.setLevel(logging.DEBUG)

SPECIAL_USE_OF_LISTS = {"FOREVER": 0,
                        "LET": 0,
                        "REPEAT": 0,
                        "IF": 0,
                        "IFELSE": 0,
                        "CAREFULLY": 0,
                        "EVERYONE": 0,
                        "DOLIST": 1,
                        "DOTIMES": 1,
                        "ASK": 1,
                        "LAUNCH": 0
                        }

PRIORITY = {'+': 1, '-': 1, '>': 0, '<': 0, '=': 0, '*': 2, '/': 2}

REG_DECIMAL_MARK = re.compile('(?<=\d),(?=\d)')


def try_or_log_error(f):
    def try_once(self, root, code, **kwargs):
        try:
            result = f(self, root, code, **kwargs)
        except IndexError as e:
            result = ""
            logger.warning(root)
            # logger.error(e)
        except AttributeError as e:
            result = ""
            logger.warning(root)
            # logger.error(e)
        return result
    return try_once


class TokenType(Enum):
    """
    Types of tokens.
    """
    STRING = "STRING"
    NUMBER = "NUMBER"
    VARIABLE = "VARIABLE"
    PRIMITIVE = "PRIMITIVE"
    OPERATOR = "OPERATOR"  # an operator from this list / + * - > < =
    PRIMITIVE_OR_WORD = "PRIMITIVE OR WORD"

    def __str__(self):
        return self.value


class Node(object):
    """ A tree node used for building an abstract syntax tree."""
    def __init__(self, value, parent=None):
        self.value = value
        self.children = []
        self.child_trans = []
        self.parent = parent
        self.empty_list = False
        self.empty_word = False

    def add_child(self, obj):
        self.children.append(obj)
        obj.parent = self
        logger.debug("Added NODE |{}| to ROOT |{}|.".format(obj.value, self.value))

    def peek_last(self):
        return self.children[-1]

    def pop_child(self):
        return self.children.pop()

    def pop_last_leaf_and_parent(self):
        if self.children:
            if self.children[-1].children:
                return self.children[-1].pop_last_leaf_and_parent()
            else:
                return self, self.pop_child()
        else:
            raise Exception('Node has no children')

    def __repr__(self, level=0):
        ret = "    "*level + str(self.value) + "\n"
        for child in self.children:
            ret += child.__repr__(level=level+1)
        return ret


class Primitive(object):
    """Storage of information about Logo primitives and their JS-translations."""
    def __init__(self, name, prim_type, args_num, trans, locale=None):
        self.name = name
        self.is_command = bool(int(prim_type))
        self.unknown_type = True if int(prim_type) > 1 else False
        # 1 - command, 0 - function, This is true if prim is a command (not function)
        self.more_args = False
        self.args = self.args_number(args_num)
        self.locale = []
        if locale:
            self.locale = locale
        if trans == "USERPROC":
            self.trans = 'yield* prims.userproc("{}", [{}]);'.format(self.name, " ,".join(["{}"]*self.args))
            self.locale = [name.lower()]
        else:
            self.trans = trans

    def args_number(self, value):
        if isinstance(value, int):
            return value
        if value.endswith('+'):
            self.more_args = True
            value = value[:-1]
        return int(value)

    def __repr__(self):
        return self.name


class Token(object):
    """
    Token of Logo code.
    Tokens can be of one of the types given in TokenType enum.
    """
    def __init__(self, val, env, abstract_value=False):
        self.value = val
        self.type = self.get_type(val, env, abstract_value)
        self.source = val

    def get_type(self, token, env, abstract_value):
        if not token:
            return TokenType.STRING
        if abstract_value:
            self.value = env.pull_abstract(token)
            return TokenType.PRIMITIVE
        if token.startswith(":"):
            self.value = self.value[1:].lower()
            return TokenType.VARIABLE
        if token.startswith('"'):
            self.value = self.value[1:]
            return TokenType.STRING
        if token.lower() in env.locale_reverse_index:
            self.value = env.pull_locale(token.lower())
            if token in "/+*-><=":
                return TokenType.OPERATOR
            return TokenType.PRIMITIVE
        try:
            self.value = int(token)
            return TokenType.NUMBER
        except ValueError:
            try:
                token_float = REG_DECIMAL_MARK.sub('.', token)
                self.value = float(token_float)
                return TokenType.NUMBER
            except ValueError:
                if token.lower().startswith('set'):
                    is_command, argnum, trans = 1, "1", 'yield* prims.callnameset("{}", {{}});'.format(token.lower())
                else:
                    is_command, argnum, trans = 0, "0", 'yield* prims.callname("{}")'.format(token.lower())
                env[token.lower()] = Primitive(token, is_command, argnum, trans, locale=[token.lower()])
                env.locale_reverse_index[token.lower()] = env[token.lower()]
                self.value = env[token.lower()]
                return TokenType.PRIMITIVE_OR_WORD

    def __repr__(self):
        return "{} <{}> <{}>".format(self.value, self.type, self.source)


class Env(dict):
    """An environment: a dict of {'var': val} pairs, with an outer Env."""

    def __init__(self, locale="English", parms=(), args=(), outer=None, **kwargs):
        super().__init__(**kwargs)
        self.update(zip(parms, args))
        self.outer = outer
        self.object_names = {}
        self.locale = locale
        logger.info("Locale: {}".format(locale))
        self.locale_reverse_index = {}
        self.with_many_args = set()
        self.read_primitives()
        self.read_localizations()
        self.STRINGS = self.read_translations(self.locale)
        self.TRANSLATIONS = self.to_english()

    # TODO: check if we need outer
    def find(self, var):
        """Find the innermost Env where var appears."""
        if var in self:
            return self
        if self.outer is None:
            return None
        return self.outer.find(var)

    def find_locale(self, var):
        """Find the innermost Env where var appears."""
        if var in self.locale_reverse_index:
            return self.locale_reverse_index
        if self.outer is None:
            return None
        return self.outer.find_locale(var)

    def parse_synonyms(self, syns):
        if syns.startswith('['):
            return syns[1:-1].split()
        else:
            return [syns]

    def read_primitives(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # read primitives from file, first line contains headers
        with open(os.path.join(BASE_DIR, 'primitives.csv'), 'r', encoding='utf-8') as prim_file:
            lines = prim_file.readlines()[1:]
        for line in lines:
            line = line.strip().split(',', maxsplit=3)
            self[line[0]] = Primitive(*line)
            if self[line[0]].more_args:
                self.with_many_args.add(self[line[0]].name)
        # print(self)

    def read_localizations(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # read localizations from file, first line contains headers
        with open(os.path.join(BASE_DIR, self.locale, 'prims.txt'), 'r', encoding='utf-16') as locale_file:
            lines = locale_file.readlines()
        for line in lines:
            if line.strip() == '':
                continue
            name, synonyms = line.strip().split(maxsplit=4)[3:]  # TODO: нужны ли примитивы с точками?
            if name.upper() in self:
                for locale_variant in self.parse_synonyms(synonyms):
                    self[name.upper()].locale.append(locale_variant.lower())
                    self.locale_reverse_index[locale_variant.lower()] = self[name.upper()]

        # check that every primitive has locales
        for primitive in self:
            if not self[primitive].locale:
                logger.error('Locale for primitive "{}" not found in {}/prims.txt.'.format(primitive, self.locale))
                raise Exception('Locale for primitive "{}" not found in {}/prims.txt.'.format(primitive, self.locale))

    def read_translations(self, locale):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(BASE_DIR, locale, 'strings.txt'), 'r', encoding='utf-16') as locale_file:
            text = locale_file.read()
        trans = defaultdict(list)
        cur_text = ''
        key = ''
        inside_bars = False
        after_linebreak = True
        for sym in text:
            if inside_bars:
                if sym == "|":
                    inside_bars = False
                    trans[key].append(cur_text.strip())
                    cur_text = ""
                else:
                    cur_text += sym
            else:
                if sym == "|":
                    inside_bars = True
                elif sym == '[' or sym == ']' or sym == '\r':
                    pass
                elif sym == '\n':
                    if cur_text.strip():
                        trans[key].append(cur_text.strip())
                    after_linebreak = True
                    cur_text = ''
                elif re.match('\s', sym):
                    if after_linebreak:
                        after_linebreak = False
                        key = cur_text
                        cur_text = ''
                    else:
                        if cur_text.strip():
                            trans[key].append(cur_text.strip())
                        cur_text = ''
                else:
                    cur_text += sym
        if cur_text.strip():
            trans[key].append(cur_text.strip())
        return trans

    def to_english(self):
        if self.locale != 'English':
            ENGSTRINGS = self.read_translations('English')
            TRANSLATIONS = {item: ENGSTRINGS[key][0] for key, values in self.STRINGS.items() for item in values}
        else:
            TRANSLATIONS = {item: item for key, values in self.STRINGS.items() for item in values}
        return TRANSLATIONS

    def pull_abstract(self, el):
        this_env = self.find(el)
        if this_env:
            return this_env[el]
        return None

    def pull_locale(self, el):
        this_env = self.find_locale(el)
        if this_env:
            return this_env[el]
        return None


class Parser(object):
    def __init__(self, locale="English"):
        self.ENV = Env(locale=locale)

        END = '(?:{})'.format('|'.join([''.join(['[{}{}]'.format(letter.upper(), letter.lower()) for letter in word])
                                      for word in self.ENV.STRINGS["sym_end"]]))
        START = '(?:{})'.format('|'.join([''.join(['[{}{}]'.format(letter.upper(), letter.lower()) for letter in word])
                                      for word in self.ENV.STRINGS["sym_to"]]))
        self.REG_PROC = re.compile("(?:^|\n)(?:\\s*)\\b{}\\b.*?\\b{}\\b".format(START, END), flags=re.DOTALL)

        self.CASHED = {}

    def make_procsarray(self, procs):
        """Receives text with procedures and comments, extracts the procedure code and cuts comments."""
        procs = self.cut_comments(procs)
        if procs.endswith("\x00"):
            procs = procs[:-1]
        procsarray = [p.strip() for p in self.REG_PROC.findall(procs.strip())]
        # print(procsarray)
        return procsarray

    def cut_definition(self, proc: str) -> str:
        """
        Cut the procedure definition, all arguments and end statement from the code string.

        Example: "TO PROC_NAME :X :Y some code END" would return "some code".
        """
        code, is_proc = [], False
        proc = re.sub('\s', ' ', proc)  # replace all space-like symbols with one space
        parts = [i for i in proc.strip().split(' ') if i]  # split by spaces and skip empty strings
        if parts[0].lower() in self.ENV.STRINGS['sym_to']:  # TODO: make sure that input is always a procedure
            is_proc = True
            parts = parts[2:]

        if parts[-1].lower() in self.ENV.STRINGS['sym_end'] and is_proc:
            parts = parts[:-1]

        if parts:
            ind = 0
            while parts[ind].startswith(':'):
                ind += 1

            result = ' '.join(parts[ind:])
        else:
            result = ''   # если процедура пустая
        logger.debug('Procedure code without procedure boilerplate: {}'.format(result))
        return result

    def cut_comments(self, chars: str) -> str:
        """Cut comments from the given Logo code."""
        if ';' not in chars:
            return chars

        new_chars = ""

        inside_bars = False
        inside_comment = False

        for char in chars:
            if inside_bars:
                if char == "|":
                    inside_bars = False
                new_chars += char
            elif inside_comment:
                if char == "\n":
                    inside_comment = False
                    new_chars += char
            else:
                if char == "|":
                    new_chars += char
                    inside_bars = True
                elif char == ";":
                    inside_comment = True
                else:
                    new_chars += char
        return new_chars

    def tokenize(self, chars: str, add_parentheses=False) -> list:
        """Convert a string of characters into a list of tokens."""

        logger.debug("Start tokenizing.")
        chars = re.sub('\s', ' ', chars.strip())
        if chars[0] != "[":
            chars = "[" + chars + "]"

        tokens = []
        special_symbols = set("[]")
        if add_parentheses:
            special_symbols.update(['(', ')'])

        current_string = ""
        inside_bars = False

        for char in chars:
            if inside_bars:
                if char == "|":
                    inside_bars = False
                    tokens.append(current_string)
                    current_string = ""
                else:
                    current_string += char
            else:
                if char == "|":
                    inside_bars = True
                elif re.match('\s', char):
                    if current_string:
                        if current_string[-1] == '"':
                            current_string += char
                        else:
                            tokens.append(current_string)
                            current_string = ""
                elif char in special_symbols:
                    if current_string:
                        tokens.append(current_string)
                        current_string = ""
                    tokens.append(char)
                else:
                    current_string += char
        if current_string:
            tokens.append(current_string)
        logger.debug("Tokens: {}".format(str(tokens)))
        return tokens

    def read_from_tokens(self, tokens: list, add_parentheses=False):
        """Read an expression from a sequence of tokens."""
        if len(tokens) == 0:
            raise SyntaxError('unexpected EOF')
        token = tokens.pop(0)
        if token == '[':
            L = []
            while tokens[0] != ']':
                L.append(self.read_from_tokens(tokens))
            tokens.pop(0)  # pop off ']'
            return L
        elif token == ']':
            raise SyntaxError('unexpected ]')
        elif token == '(':
            L = []
            while tokens[0] != ')':
                L.append(self.read_from_tokens(tokens))
            tokens.pop(0)  # pop off ')'
            return tuple(L)
        elif token == ')':
            raise SyntaxError('unexpected )')
        else:
            return Token(token, self.ENV)

    def collect_code(self, tokens):
        code = []
        for token in tokens:
            if isinstance(token, list):
                code.append('[{}]'.format(self.collect_code(token)))
            elif isinstance(token, tuple):
                code.append('({})'.format(self.collect_code(token)))
            else:
                if any(i in token.source for i in '[] '):
                    token.source = '|{}|'.format(token.source)
                code.append(token.source)
        return ' '.join(code)

    def process_list_of_words(self, words):
        if not words:
            return None
        code = self.collect_code(words)
        if not code:
            return None
        tokens = self.tokenize(code, add_parentheses=False)
        tree = self.read_from_tokens(tokens, add_parentheses=False)
        for ind, el in enumerate(tree):
            if isinstance(el, list):
                if el:
                    tree[ind] = self.process_list_of_words(el)
                else:
                    tree[ind] = []
            elif el.type in {TokenType.PRIMITIVE_OR_WORD, TokenType.PRIMITIVE, TokenType.OPERATOR}:
                el.type = TokenType.STRING
                el.value = el.source
                logger.debug("Reinterpreing <{}> as string.".format(el.value))
        return tree

    def process_list_of_instructions(self, instructions):
        if not instructions:
            return None
        code = self.collect_code(instructions)
        tokens = self.tokenize(code, add_parentheses=True)
        tree = self.read_from_tokens(tokens, add_parentheses=True)
        return tree

    def check_tree(self, root, code, older_q, **kwargs):
        if 'all_constants' in kwargs and kwargs['all_constants']:
            code = self.process_list_of_words(code)
            logger.debug("Updated tree: list of words. Root node: {}. Code: {}".format(root.value, code))
            node = Node("LIST")
            root.add_child(node)
            if code:
                self.create_tree(node, code, older_q=older_q, check=False)
            else:
                node.empty_word = True
                logger.debug("This List represents an empty list of words.")
        elif isinstance(code, list):
            node = Node("LIST")
            if isinstance(root.value, Token) and root.value.value.name in SPECIAL_USE_OF_LISTS:
                PREV_NUM_OF_NODES = SPECIAL_USE_OF_LISTS[root.value.value.name]
                if len(root.children) >= PREV_NUM_OF_NODES:
                    code2 = code
                    code = self.process_list_of_instructions(code)

                    # this piece adds original logo-code (translated as logo constant) to js-code
                    if root.value.value.name in {'FOREVER', 'LAUNCH'}:
                        code2 = self.process_list_of_words(code2)
                        root2, node2 = Node('ROOT'), Node("LIST")
                        root2.add_child(node2)
                        tree2 = self.create_tree(node2, code2, older_q=None, check=False)
                        trans = self.post_order_translation(tree2)
                        # logger.debug('translation ' + trans)
                        root.value.value.trans = root.value.value.trans[:-2] + ', {});'.format(trans)

                    logger.debug("Updated tree: special use of lists. Root node: {}. Code: {}".format(root.value, code))
                    if not code:
                        node.empty_list = True
                        logger.debug("This List represents an empty list of instructions.")
                else:
                    code = self.process_list_of_words(code)
                    logger.debug("Updated tree: list of words. Root node: {}. Code: {}".format(root.value, code))
                    if not code:
                        node.empty_word = True
                        logger.debug("This List represents an empty list of words.")
            elif isinstance(root.value, Token) and root.value.value.name not in SPECIAL_USE_OF_LISTS:
                code = self.process_list_of_words(code)
                logger.debug("Updated tree: list of words. Root node: {}. Code: {}".format(root.value, code))
                if not code:
                    node.empty_word = True
                    logger.debug("This List represents an empty list of words.")
            root.add_child(node)
            if code:
                self.create_tree(node, code, older_q=older_q, check=False)
        elif isinstance(code, tuple):
            node = Node("TUPLE")
            root.add_child(node)
            self.create_tree(node, code, older_q=older_q, check=False)
        return root

    # @try_or_log_error
    def create_tree(self, root: Node, code, older_q=None, check=True, **kwargs) -> Node:
        logger.debug("Create tree with root node: {}. Code: {}".format(root.value, code))
        # logger.debug("Older_q: {}".format(older_q))
        if check and isinstance(code, (tuple, list)):
            return self.check_tree(root, code, older_q, **kwargs)

        try:
            queue = deque(code)
        except TypeError:
            queue = deque([code])

        new_args_number = None
        if root.value == "TUPLE":
            if isinstance(queue[0], Token) and queue[0].type == TokenType.PRIMITIVE \
                    and queue[0].value.name in self.ENV.with_many_args:
                new_args_number = len(queue) - 1
                logger.info("Primitive <{}> has {} arguments.".format(queue[0].value.name, new_args_number))

        while queue:
            el = queue.popleft()
            # logger.debug("Processing el: {}".format(el))
            if isinstance(el, list):
                node = Node("LIST")
                root.add_child(node)
                self.create_tree(node, el, older_q=queue, check=False)
            elif isinstance(el, tuple):
                node = Node("TUPLE")
                root.add_child(node)
                self.create_tree(node, el, older_q=queue, check=False)
            else:
                node = Node(el)
                if el.type == TokenType.PRIMITIVE_OR_WORD:
                    el.type = TokenType.PRIMITIVE
                    logger.warning('Interpret unknown token <{}> as primitive.'.format(el.value))
                if el.type == TokenType.PRIMITIVE:
                    if not new_args_number:
                        args_number = el.value.args
                        ignore_error = True
                    else:
                        args_number = new_args_number
                        ignore_error = False
                    while len(node.children) < args_number:
                        try:
                            self.create_tree(node, queue.popleft(), older_q=queue)
                        except IndexError:
                            if ignore_error:
                                logger.debug("PRIM INDEX ERROR el: {}, type: {},"
                                             "args: {}, older_q: {}".format(el.value, el.type,
                                                                            el.value.args, older_q))
                                self.create_tree(node, older_q.popleft(), older_q=queue if queue else older_q)
                            else:
                                break
                    root.add_child(node)
                elif el.type == TokenType.OPERATOR:

                    logger.debug("THIS OPERATOR <{}>".format(el.value.name))
                    logger.debug("THIS ROOT <{}>".format(root))

                    cur_root = root
                    prev_node = root.peek_last()
                    # logger.debug("PREV_NODE <{}>".format(prev_node))
                    while True:
                        if prev_node.value == "TUPLE":
                            operand = cur_root.pop_child()
                            node.add_child(operand)
                            cur_root.add_child(node)
                            break
                        elif prev_node.value == "LIST":
                            cur_root = prev_node
                            prev_node = prev_node.peek_last()
                        else:  # Token
                            if prev_node.value.type == TokenType.OPERATOR:
                                prev_op = prev_node.value.value.name
                                this_op = el.value.name
                                if PRIORITY[prev_op] >= PRIORITY[this_op]:
                                    operand = cur_root.pop_child()
                                    node.add_child(operand)
                                    cur_root.add_child(node)
                                    break
                                else:
                                    cur_root = prev_node
                                    prev_node = prev_node.peek_last()
                            elif prev_node.value.type == TokenType.PRIMITIVE and prev_node.value.value.args > 0:
                                cur_root = prev_node
                                prev_node = prev_node.peek_last()
                            else:  # STRING, NUMBER, VARIABLE, all PRIMITIVES with arguments
                                operand = cur_root.pop_child()
                                node.add_child(operand)
                                cur_root.add_child(node)
                                break
                    try:
                        self.create_tree(node, queue.popleft(), older_q=queue)
                    except IndexError:
                        logger.debug("INFS INDEX ERROR older_q: {}".format(older_q))
                        self.create_tree(node, older_q.popleft(), older_q=older_q)
                else:
                    root.add_child(node)
        # logger.debug("Current tree:\n{}".format(root))
        return root

    def parse(self, code: str, **kwargs) -> Node:
        # tokenize
        tokens1 = self.tokenize(code, add_parentheses=True)
        tree = self.read_from_tokens(tokens1, add_parentheses=True)
        logger.debug("Listed tree structure: {}".format(str(tree)))
        # create tree with an artificial root node
        root1 = Node("ROOT")
        logger.debug("Now creating tree...")
        new_tree = self.create_tree(root1, tree, **kwargs)
        logger.debug("Tree created.")
        logger.debug("Tree:\n{}".format(new_tree))
        return new_tree

    def post_order_translation(self, root: Node):
        """Post order traversal of the tree with translation."""
        child_translations = []
        for child in root.children:
            child_translations.append(self.post_order_translation(child))
        root.child_trans = child_translations
        return self.translate(root)

    def translate_list(self, list_node: Node) -> str:
        # logger.debug("LIST NODE CHILDREN: {}".format(list_node.children))
        if len(list_node.child_trans) == 0:
            if list_node.empty_word:
                return '""'
            if list_node.empty_list:
                return ''

            return '""'
        elif len(list_node.child_trans) == 1:
            el = list_node.child_trans[0]
            if isinstance(list_node.children[0].value, Token) and \
                    list_node.children[0].value.type in {TokenType.STRING, TokenType.NUMBER, TokenType.VARIABLE} and \
                    list_node.parent.value != "ROOT":
                parent = list_node.parent.value
                if isinstance(parent, Token) and isinstance(parent.value, Primitive) and \
                        parent.value.name in {'ASK', 'IFELSE'}:
                    return '{}'.format(el)
                return '[{}]'.format(el)
            return el
        else:
            parent_node_token = list_node.parent.value
            special_parent_case = False
            if isinstance(parent_node_token, Token):
                if parent_node_token.type == TokenType.PRIMITIVE and parent_node_token.value.name == "LET":
                    result = []
                    for ind, el in enumerate(list_node.child_trans):
                        if ind % 2 == 0:
                            name = re.search('".*?"', list_node.child_trans[ind])
                            if name is None:
                                name = list_node.child_trans[ind]
                            else:
                                name = name.group(0)
                            result.append(parent_node_token.value.trans.format(name,
                                                                               list_node.child_trans[ind + 1]))
                    return ' '.join(result)
                if parent_node_token.type == TokenType.PRIMITIVE:
                    if parent_node_token.value.name not in SPECIAL_USE_OF_LISTS:
                        return '[{}]'.format(', '.join(list_node.child_trans))
                    else:
                        if list_node.parent.children.index(list_node) < SPECIAL_USE_OF_LISTS[parent_node_token.value.name]:
                            return '[{}]'.format(', '.join(list_node.child_trans))
                        else:
                            special_parent_case = True

            if all(i.endswith('"') and i.startswith('"') for i in list_node.child_trans):
                return '[{}]'.format(', '.join(list_node.child_trans))

            command_list = []
            for el in list_node.child_trans:
                # if not (el.strip().endswith("}") or el.strip().endswith(";")):
                if el.endswith(')'):
                    el += ';'
                command_list.append(el)

            if all(i.endswith(';') or i.endswith('}') for i in command_list):
                return ' '.join(command_list)

            return '[{}]'.format(', '.join(command_list))

    def update_child_translations(self, root):
        for child_num, child in enumerate(root.children):
            if isinstance(child.value, Token) and isinstance(child.value.value, Primitive):
                primitive = child.value.value
                if primitive.unknown_type:
                    cur_trans = root.child_trans[child_num]
                    if cur_trans.endswith(';'):
                        root.child_trans[child_num] = cur_trans[:-1]
                if primitive.is_command:
                    logger.warning('Primitive {} is a command, but is used as an argument.'.format(primitive.name))
        return root

    def translate(self, root: Node) -> str:
        """Translate a node to JS-code."""
        token = root.value
        if token == "ROOT":
            return "\n".join(root.child_trans)
        if token == "LIST" or token == "TUPLE":
            result = self.translate_list(root)
            # logger.debug('LIST TRANSLATE RESULT: {}'.format(result))
            return result
        if token.type == TokenType.PRIMITIVE or token.type == TokenType.OPERATOR:
            if token.value.args == 0:
                if len(root.child_trans) == 0:
                    return token.value.trans
                else:
                    # TODO: fix these exceptions because we have translate from list
                    raise Exception('Primitive {} must has 0 arguments, {} given.'.format(token.value.name,
                                                                                          len(root.child_trans)))
            else:
                if token.value.name == "LET":
                    return root.child_trans[0]
                if token.value.name == "SAYAS":
                    name = re.search('".*?"', root.child_trans[1])
                    if name is None:
                        name = root.child_trans[1]
                    else:
                        name = name.group(0)
                    return token.value.trans.format(root.child_trans[0], name)
                if token.value.name in self.ENV.with_many_args:
                    return token.value.trans.format(', '.join(root.child_trans))
                if len(root.child_trans) == token.value.args:
                    root = self.update_child_translations(root)
                    return token.value.trans.format(*root.child_trans)
                else:
                    raise Exception('Primitive {} must have {} arguments, {} given.'.format(token.value.name,
                                                                                            token.value.args,
                                                                                            len(root.child_trans)))
        elif token.type == TokenType.STRING:
            if root.parent.value == 'LIST':
                if '"' in token.source:
                    return '"{}"'.format(token.source.replace('"', '\\"'))
            return '"{}"'.format(token.value.replace('"', '\\"'))
        elif token.type == TokenType.NUMBER:
            return str(token.value)
        elif token.type == TokenType.VARIABLE:
            return 'scope.thing("{}")'.format(token.value)
        raise Exception("Unknown token type: {}".format(token.type))

    def check_tail_recursion(self, tree, name, args_str):
        root = tree
        tree = tree.children[0]
        if tree.children:
            last_call = tree.children[-1].value
            if isinstance(last_call, Token) and isinstance(last_call.value, Primitive):
                if last_call.value.name == name:
                    args = [self.post_order_translation(node) for node in tree.children[-1].children]
                    tail_recursion_code = """
                            while (true) {{{{
                                {{}}
                                scope.reassign([{}], [{}]);
                            }}}}
                            """.format(args_str, ', '.join(args))
                    tree.children.pop()
                    return root, tail_recursion_code
        return root, '{}'

    def check_tree_correct(self, tree, **kwargs):
        # logger.debug(kwargs)
        all_constants = kwargs['all_constants'] if 'all_constants' in kwargs else False
        if not all_constants:
            token_types = {TokenType.PRIMITIVE, TokenType.OPERATOR}
            for child in tree.children:
                if not (isinstance(child.value, Token) and child.value.type in token_types):
                    # logger.debug('{} {}'.format(child.value, child.value.type))
                    return False
        return True

    def translate_from_anywhere(self, proc_text, **kwargs):
        proc_text = self.cut_comments(proc_text)
        new_code = self.cut_definition(proc_text)
        logger.debug("Start parsing Logo code...")
        if new_code:
            tree = self.parse(new_code, **kwargs)
            if 'proc_name' in kwargs:
                args_str = ', '.join(kwargs['args_list']) if 'args_list' in kwargs else ""
                tree, template = self.check_tail_recursion(tree, kwargs['proc_name'], args_str)
            else:
                template = '{}'
            correct_tree = self.check_tree_correct(tree.children[0], **kwargs)
            logger.debug("Start translating...")
            trans = self.post_order_translation(tree)
            trans = template.format(trans)
            if not correct_tree:
                logger.critical('\n!!!\nCannot correctly translate logo code.\nLogo code: {}.\nFailed result: {}\n!!!'.format(new_code, trans))
                raise Exception('Could not correctly translate logo code.')
        else:
            trans = ''
        return trans

    def translate_from_project(self, proc_text: str, **kwargs) -> str:
        wrapper = kwargs['wrapper'] if 'wrapper' in kwargs else ''
        args_list = kwargs['args_list'] if 'args_list' in kwargs else None
        # wrapper="", args_list=None, all_constants=False

        # Интерфейс глобального логокода (событий):
        global_logo_code = """
            function* (globals, processContext, obj){{
                var prims = processContext.prims;
                var vars = globals.vars;
                var userproc = globals.userproc;
                var scope = vars.globalScope;
                {}
                }}
            """
        # Объявление пользовательской процедуры:
        user_procedure = """
                function* (args){{
                    var scope = vars.newLocalScope();
                    scope.setArgs([{}],args);
                    {}
                    }}
                    """
        trans = self.translate_from_anywhere(proc_text, **kwargs)
        if wrapper == "GLOBAL":
            trans = global_logo_code.format(trans)
        if wrapper == "USERPROC":
            if args_list is None:
                args_list = []
            trans = user_procedure.format(', '.join(args_list), trans)
        trans = beautify(trans)

        logger.info("Procedure code: {}".format(proc_text.replace('\n', '    ')))
        logger.info("JS code: \n{}".format(trans))
        return trans

    def primary_analysis(self, proc):
        logger.debug('Procedure code: {}'.format(proc.replace('\n', '    ')))
        proc = self.cut_comments(proc)
        proc = re.sub('\s', ' ', proc)
        parts = proc.split(' ')  # make sure no successive spaces
        if parts[0].lower() not in self.ENV.STRINGS["sym_to"]:  # TODO: is this always proc??
            # return "", 0, []
            raise Exception('Procedure definition should start with TO')
        proc_name = parts[1]  # TODO: if no spaces are allowed in names
        arg_num = 0
        args_list = []
        for el in parts[2:]:
            if el.startswith(':'):
                arg_num += 1
                args_list.append(el)
            else:
                break
        logger.debug('Procedure name: {}, Arg_num: {}, Args_list: {}'.format(proc_name, arg_num, args_list))
        return proc_name, arg_num, args_list


if __name__ == "__main__":
    # парсим аргументы командной строки
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('input', nargs='?', type=str, default="stdin",
                   help="Путь к входному файлу с лого-кодом. Если имя файла опущено, или если вместо имени "
                        "указано \"-\", то будет использоваться стандартный ввод.")
    p.add_argument('output', nargs='?', type=str, default="stdout",
                   help="Путь к выходному файлу с лого-кодом. Если имя файла опущено, или если вместо имени "
                        "указано \"-\", то будет использоваться стандартный вывод.")
    p.add_argument('--locale', type=str, dest="locale", default="English",
                   help="Код языка локализации: en или ru. По умолчанию этот аргумент принимает значение \"en\", "
                        "при запуске на английских проектах --locale можно не прописывать. Но для трансляции проектов "
                        "с русскими командами нужно будет прописать --locale ru.")
    p.add_argument('--log', type=str, dest="log", default="INFO",
                   choices=["debug", "info", 'warning', 'error', 'critical'],
                   help="Наименьший уровень сообщений, которые необходимо распечатывать:\n"
                        "  CRITICAL - печатать сообщение, только когда случилась серьезная ошибка;\n"
                        "  ERROR- печатать сообщение, только когда случилась любая ошибка;\n"
                        "  WARNING - печатать предупреждения о возможной ошибке.\n"
                        "  INFO - печатать информацию о текущем процессе;\n"
                        "  DEBUG - подробно распечатывать результат каждого этапа работы программы\n"
                        "          (в частности, на этом уровне можно проверять абстрактное синтаксическое дерево,\n"
                        "          которое строит программа). Значением по умолчанию этого аргумента является INFO.")

    cmd_args = p.parse_args()

    # для целей тестирования можно запускать этот код без командной строки,
    # тогда нужно установить test=True и нужные аргументы нужно прописать ниже:
    test = False
    if test:
        class Object(object):
            pass
        cmd_args = Object()
        vars(cmd_args).update(input="utils/logo_code.txt", output="stdout", locale="Russian", log="debug")

    # устанавливаем уровень логгирования
    LOG_LEVEL = logging.getLevelName(cmd_args.log.upper())
    logging.basicConfig(level=LOG_LEVEL)
    logging.getLogger('LogoParser').setLevel(LOG_LEVEL)

    PARSER = Parser(locale=cmd_args.locale.lower())

    logo_code = ""
    if cmd_args.input == 'stdin' or cmd_args.input == '-':
        logging.info('Input your logo code:')
        line = input()
        while line:
            logo_code += line + '\n'
            line = input()
    else:
        logging.info('Reading your logo code from {}.'.format(cmd_args.input))
        with open(cmd_args.input, 'r', encoding='utf-8') as f:
            logo_code = f.read()
    # todo: parse multiple procs or message that you do not accept procs

    trans = PARSER.translate_from_anywhere(logo_code)
    trans = beautify(trans)

    if cmd_args.output == "stdout":
        logging.info("Translated JS code:\n{}".format(trans))
    else:
        logging.info('Writing translated JS code to {}.'.format(cmd_args.output))
        with open(cmd_args.output, 'w', encoding='utf-8') as f:
            f.write(trans)