# -*- coding: utf-8 -*-

"""
Parser for MicroWorlds project.

Parser can process MWX and MJ3 projects.
The contents of the project are transferred to data.js file.
"""

__author__ = "Elmira Mustakimova <cclikespizza@gmail.com>"
__version__ = "1.19.5"

import argparse
import base64
import io
import json
import logging
import os
import re
import string
import struct
import sys
import uuid
import zlib

from rtf import striprtf, get_function_for_rtf_processing
from collections import deque
from LogoParser import Primitive, Parser

logger = logging.getLogger("ProjectParser")
logger.setLevel(logging.DEBUG)

REG_SQUARE_BRACKETS = re.compile("\[(.*?)\]")
FORMATTER = string.Formatter()


class Object:
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4, ensure_ascii=False)


def read_object_rules(fname):
    """
    Читаем правила перевода объектов проекта, создаем словарь правил вида TYPE: LIST OF RULES.

    В файле с правилами есть заголовок. Каждая строка вида
    тип объекта, шаблон примитива, команда или датчик, число аргументов, шаблон перевода.
    Разделитель табуляция.
    """
    types = {}
    logger.info("Reading object rules: {}".format(fname))
    with open(os.path.join(BASE_DIR, fname), 'r', encoding='utf-8') as rules:
        lines = rules.readlines()[1:]
    for line in lines:
        line = line.strip().split('\t', maxsplit=4)
        type_name = line[0]
        line[2], line[3] = int(line[2]), int(line[3])
        if type_name in types:
            types[type_name].append(line[1:])
        else:
            types[type_name] = [line[1:]]
    logger.info("Object rules loaded.")
    return types


def add_new_prim(new_prim, is_command, argnum, new_trans, type_name):
    """Add new primitives with object names to the dict of primitives."""
    if new_prim not in PARSER.ENV:
        PARSER.ENV[new_prim] = Primitive(new_prim, is_command, argnum, new_trans)
        PARSER.ENV.locale_reverse_index[new_prim] = PARSER.ENV[new_prim]
        logger.info("Object rule for <{}>  of type <{}> is added to dictionary. Rule: <{}>".format(new_prim,
                                                                                                   type_name,
                                                                                                   new_trans))
    else:
        logger.warning("Object rule for <{}> is already in dictionary. Rule: <{}>".format(new_prim, new_trans))


def process_rules(type_name, object_name):
    """Update rules of given type with the given name."""
    object_name = object_name.lower()
    rules = TYPES[type_name]
    for rule in rules:
        is_command, argnum = rule[1], rule[2]
        new_trans = rule[-1].format(name=object_name)
        keys = [i[1] for i in FORMATTER.parse(rule[0]) if i[1] is not None and i[1] != 'name']
        if len(keys) == 0:
            new_prim = rule[0].format(name=object_name)
            add_new_prim(new_prim, is_command, argnum, new_trans, type_name)
        elif len(keys) == 1:
            translations = PARSER.ENV.STRINGS[keys[0]]
            for tr in translations:
                d = {'name': object_name, keys[0]: tr}
                new_prim = FORMATTER.vformat(rule[0], [], d)
                add_new_prim(new_prim, is_command, argnum, new_trans, type_name)
        else:
            raise Exception('Too many template strings in object_rules.')


def open_dll_head(fhead_name):
    """
    Читает файл с оглавлением library.lld и создает словарь вида {Хэш_32 : оффсет}.
    В library.lld хранятся хэши битмэпов и оффсеты, по которым можно найти соответствующий битмэп в library.dll.
    """
    dll_head1 = {}
    logger.info("Reading bitmap index: {}".format(fhead_name))
    with open(fhead_name, 'rb') as fhead:
        while 1:
            off = fhead.read(34).decode('utf-16')
            if len(off) == 0:
                break
            key = fhead.read(66).decode('utf-16')
            off = struct.unpack("i", fhead.read(4))[0]
            dll_head1[key] = off
    logger.info("Bitmap index loaded.")
    return dll_head1


def read_mwx_content(fname):
    """Читает файл проекта MWX, распаковывает его и возвращает содержимое."""
    logger.info("Unpacking mwx-project: {}".format(fname))
    with open(fname, mode='rb') as file:
        fileContent = file.read()
        mw = fileContent.find(b'\x00m\x00w')
        space = fileContent[mw:].find(b' \x00') + mw
        dot = fileContent[space:].find(b'.\x00') + space
        language = fileContent[space:dot].decode('utf-16').replace('\x00', '').strip()

        start = fileContent.find(b'[')
        end = fileContent[start:].find(b']') + start

        projectsize = fileContent[start + 2:end].decode("utf-16")

        version_PL = b'version|PL' in fileContent[end: end + 100].replace(b'\x00', b'')

        if version_PL:
            data = zlib.decompress(fileContent[184 + len(projectsize) * 2:])
        else:
            data = zlib.decompress(fileContent[124 + len(projectsize) * 2:])
    logger.info("Project {} unpacked.".format(fname))
    return language, version_PL, data


def add_proc_to_dict(procs):
    """Добавляет имена процедур в окружение примитивов."""
    procsarray = PARSER.make_procsarray(procs)

    for proc in procsarray:
        proc_name, arg_num, args_list = PARSER.primary_analysis(proc)
        if proc_name:
            PARSER.ENV[proc_name] = Primitive(proc_name, 2, arg_num, "USERPROC", locale=[proc_name, proc_name.lower()])
            PARSER.ENV.locale_reverse_index[proc_name.lower()] = PARSER.ENV[proc_name]
            logger.info("Rule for user procedure <{}>  is added to dictionary.".format(proc_name))


def get_numbers(content: str):
    """
    Возвращает массив чисел в виде строк из записи вида [\d \d+]
    (внутри квадратных скобок любое количество чисел через пробел).
    """
    return REG_SQUARE_BRACKETS.search(content).group(1).split()


def temporary_file_length(projname, tempname):
    """Читает файл проекта, записывает распакованное содержимое во временный файл и возвращает длину содержимого."""
    language, version, mwx_content = read_mwx_content(projname)
    version = 'PERVOLOGO' if version else 'MWX'
    with open(tempname, 'wb') as fname:
        fname.write(mwx_content)
    return language, version, len(mwx_content)


def get_key_values(f):
    """Читает из файла C struct с числом, возвращает байтовую строку из соответствующего числа байтов."""
    off = f.read(4)
    off = struct.unpack("<i", off)[0] - 4
    content = f.read(off)
    content = struct.unpack("<{}s".format(off), content)[0]
    return content


def create_files(path, mode, content):
    fm = open(path, mode)
    logger.info("Created multimedia file {}.".format(path))
    fm.write(content)
    fm.close()
    return path


def primary_process_object(object_content):
    """
    Предварительная обработка объектов из проекта:
    - записываем в окружение имя мультимедиа, имя текстового окна, set+имя текстового окна, имя слайдера
    - создаем файл с мультимедиа
    """
    new_f = io.BytesIO(object_content)
    new_f.read(14)
    object_name, object_type = "", ""
    while new_f.tell() < len(object_content):
        content = get_key_values(new_f)
        try:
            c = content.decode('utf-16').split('\x00')
        except:
            c = content[:30].decode('utf-16').split('\x00')
        key = c[0]
        if key == "name":
            object_name = c[-1].replace('|', '')
        elif key == "type":
            object_type = c[-1]
        elif key == "data":
            wav_filename = os.path.join(BASE_DIR, FILESDIR, object_name + ".wav")
            try:
                create_files(wav_filename, "xb", content[10:])
            except:
                create_files(wav_filename, "wb", content[10:])
    if object_type in TYPES:
        process_rules(object_type, object_name)


def primary_process_object_pro(object_content):
    """
    Предварительная обработка объектов из проекта:
    - записываем в окружение имена процедур из Shapes.
    """
    new_f = io.BytesIO(object_content)
    new_f.read(20)
    object_name, object_type = "", ""
    while new_f.tell() < len(object_content):
        content = get_key_values(new_f)
        try:
            c = content.decode('utf-16').split('\x00')
        except:
            c = content[:30].decode('utf-16').split('\x00')
        key = c[0]
        if key == "name":
            object_name = c[-1].replace('|', '')
        elif key == "type":
            object_type = c[-1]
        elif key == "ShapesProcs":
            logger.info('Translate code from ShapesProcs.')
            procs = striprtf(content[38:])
            add_proc_to_dict(procs)
        elif key == "have-list":
            if object_type == "t":
                turtles_have = c[-1][4:-1]
                if turtles_have.strip():
                    tree = PARSER.parse(turtles_have).children[0]
                    turtles_vars = [el.value.source for i, el in enumerate(tree.children) if i % 2 == 0]
                    for t_var in turtles_vars:
                        process_rules("turtlevar", t_var)
    if object_type in TYPES:
        process_rules(object_type, object_name)


def primary_process_page(page, offset=10):
    """
    Предварительная обработка страниц проекта:
    - записываем в окружение имена страниц,
    - запускаем предварительную обработку объектов проекта.
    """
    new_f = io.BytesIO(page)
    new_f.read(offset)
    while new_f.tell() < len(page):
        content = get_key_values(new_f)
        try:
            c = content.decode('utf-16').split('\x00')
        except:
            c = content[:30].decode('utf-16').split('\x00')
        key = c[0]
        if key == "name":
            name = c[-1].replace('|', '').lower()
            process_rules("page", name)
        elif key == "object":
            primary_process_object(content)
        elif key == "ObjectPro":
            primary_process_object_pro(content)


def first_run():
    """Первый проход по содержимому проекта - выполняем предварительную обработку страниц и объектов."""
    logger.info("Start first run.")
    f = open(TEMPORARY_FILE, 'rb')
    f.read(32)
    while f.tell() < FILE_LEN:
        content = get_key_values(f)
        c = content[:100].decode('utf-16').split('\x00')
        key = c[0]
        # todo: object-rules - projectvar & turtlevar
        if key == "procedures":
            logger.info('Translate code from procedures.')
            procs = striprtf(content[34:])
            add_proc_to_dict(procs)
        elif key == "page":
            primary_process_page(content)
        elif key == "globals":
            global_vars = content.decode('utf-16')[9:-1]
            if global_vars.strip():
                projectvars = PARSER.tokenize(global_vars)[1:-1]
                for projectvar in projectvars:
                    process_rules("projectvar", projectvar)
        elif key == "turtles-have":
            turtles_have = content.decode('utf-16')[14:-1]
            if turtles_have.strip():
                turtles_vars = PARSER.tokenize(turtles_have)[1:-1]
                for t_var in turtles_vars:
                    process_rules("turtlevar", t_var)
        elif key == "templatepage":
            primary_process_page(content, offset=26)
    f.close()
    logger.info("First run finished.")


def process_lists(lists_text):
    lists = []
    stack = deque()
    current_list = ''
    for symbol in lists_text:
        if symbol == '[':
            stack.append(symbol)
        elif symbol == ']':
            stack.pop()
        current_list += symbol
        if not stack:
            current_list = current_list.strip()
            if current_list:
                lists.append(current_list)
                current_list = ''
    return lists


def process_procedures(obj_array, procsarray):
    """Выделяет процедуры, переводит их в JavaScript и добавляет код в массив процедур."""
    logger.info("Processing procedures...")
    if not procsarray:
        logger.info("No procedures found.")

    for ind, proc in enumerate(procsarray):
        proc_name, arg_num, args_list = PARSER.primary_analysis(proc)
        if proc_name and proc_name not in PARSER.ENV:
            PARSER.ENV[proc_name] = Primitive(proc_name, 1, arg_num, 'USERPROC', locale=[proc_name])
            PARSER.ENV.locale_reverse_index[proc_name.lower()] = PARSER.ENV[proc_name]
            logger.info("Rule for user procedure <{}>  is added to dictionary.".format(proc_name))

        proc_obj = Object()
        args_list = ['"{}"'.format(i.replace(':', '')) for i in args_list]

        logger.info("Procedure {}: {}".format(ind+1, proc_name))
        logger.info("Number of arguments: {}. "
                    "List of arguments: {}".format(len(args_list),
                                                   ' '.join(args_list) if args_list else "empty"))
        logger.info('Translate code from procedures.')
        code = PARSER.translate_from_project(proc, wrapper="USERPROC", args_list=args_list, proc_name=proc_name) \
            if proc not in ['', '||'] else ""

        proc_obj.code = code.strip()
        proc_obj.name = proc_name
        obj_array.append(proc_obj)


def process_global_shapes(shapes, obj):
    """Обработка форм."""
    dll_head = open_dll_head('library.lld')
    logger.info("Processing global shapes...")

    new_f = io.BytesIO(shapes)
    new_f.read(26)
    shapes_dict = {}
    while new_f.tell() < len(shapes):
        content = get_key_values(new_f)
        try:
            c = content.decode('utf-16').split('\x00')
        except:
            c = content[:30].decode('utf-16').split('\x00')
        key = c[0]
        if key == "mapshapenumber":
            values = c[1][1:-1].split()[1:]
            for ind, val in enumerate(values):
                if ind % 2 == 1:
                    if val == "[]":
                        val = ""
                    number = int(values[ind-1])
                    shapes_dict[number] = {"hash": val}
        elif key == "shapenames":
            values = ' '.join(c[1][1:-1].split()[1:])
            if values.strip():
                tree = PARSER.parse(values, all_constants=True).children[0]
                tvars = {el.value.source: PARSER.post_order_translation(tree.children[i + 1])
                         for i, el in enumerate(tree.children) if i % 2 == 0}
                for key in tvars:
                    num = int(key)
                    try:
                        value = json.loads(tvars[key])
                    except:
                        value = ''
                        logger.warning('FAILED TO READ VALUE OF shapenames')
                    if num in shapes_dict:
                        shapes_dict[num]["name"] = value
                    else:
                        shapes_dict[num] = {"hash": "", "name": value}

    obj.GlobalShapes = []
    for key in shapes_dict:
        loc_obj = Object()
        number = key
        if VERSION == 'PERVOLOGO':
            number -= 1
        loc_obj.number, loc_obj.hash, loc_obj.name = number, shapes_dict[key]["hash"], shapes_dict[key]["name"]
        obj.GlobalShapes.append(loc_obj)

        # Ниже происходит обработка библиотечных битмэпов
        if loc_obj.hash != '':
            if loc_obj.hash in dll_head:
                bmp_obj = Object()
                bmp_obj.hash = loc_obj.hash
                FBASE.seek(dll_head[bmp_obj.hash])
                data_off = FBASE.read(12)  # в 12 байтах закодированы 3 числа
                # Что за первые два числа - непонятно, но третье число указывает длину битмэпа в байтах
                data_off = struct.unpack("<3i", data_off)
                content = FBASE.read(data_off[2])
                if BUFFERFILES:
                    bmp_obj.data = ''
                    name = str(uuid.uuid4())
                    bmp_obj.datafile = create_files(os.path.join(BASE_DIR, FILESDIR, name + '.bmp'), 'wb', content)
                else:
                    bmp_obj.data = str(base64.b64encode(content))[2:-1]
                    bmp_obj.datafile = ''
                obj.Bitmaps.append(bmp_obj)


def process_bitmaps(bitmaps, this_obj):
    """Обработка битмэпов"""
    new_f = io.BytesIO(bitmaps)
    new_f.read(16)
    while new_f.tell() + 4 < len(bitmaps):
        old_cur = new_f.tell()
        content = get_key_values(new_f)
        new_cur = new_f.tell()
        try:
            c = content.decode('utf-16').split('\x00')
        except:
            c = content[:80].decode('utf-16').split('\x00')
        loc_obj = Object()
        loc_obj.hash = c[0]  # hash
        # TODO: why 80?? not len(c[0])*2+2:
        if BUFFERFILES:
            loc_obj.data = ''
            name = str(uuid.uuid4())
            loc_obj.datafile = create_files(os.path.join(BASE_DIR, FILESDIR, name + '.bmp'), 'wb', content[80:new_cur-old_cur])
        else:
            loc_obj.data = str(base64.b64encode(content[80:new_cur-old_cur]))[2:-1]  # data
            loc_obj.datafile = ''
        this_obj.Bitmaps.append(loc_obj)


def get_logo_constant(logo_code):
    code = PARSER.translate_from_project(logo_code, all_constants=True)
    # TODO give warning if could not load json
    js_code = json.loads(code) if code else ""
    return js_code


def process_object(obj_content, this_obj):
    """Обработка объектов проекта"""
    new_obj = Object()
    new_f = io.BytesIO(obj_content)
    new_f.read(14)
    while new_f.tell() < len(obj_content):
        content = get_key_values(new_f)
        try:
            c = content.decode('utf-16').split('\x00')
        except:
            c = content[:30].decode('utf-16').split('\x00')
        key = c[0]
        if key in {'locked?', 'visible?', 'snaped?', 'singleline?', 'inst', 'duration',
                   'type', 'show-name?', 'vertical?', 'name', 'volume', 'tempo', 'unicode-notes',
                   'min', 'max', 'current', 'value', 'defvalue', 'mode', 'position'}:
            attr_name = key.replace('?', '').replace('-', '_')
            vars(new_obj)[attr_name] = get_logo_constant(c[1])
            if attr_name == 'name' and new_obj.type in {'music', 'record', 'melody', 'video'}:
                # TODO: if type record\audio
                if BUFFERFILES:
                    new_obj.filename = os.path.join(BASE_DIR, FILESDIR, new_obj.name + ".wav")
                else:
                    new_obj.filename = new_obj.name + ".wav"
                # TODO: check that mode is Embeded, if empty - WARNING: Object is not used because not embeded
        elif key == 'label':
            new_obj.label = c[1]
        elif key == 'hash':
            new_obj.hash = c[1]
        elif key == "rect":
            tmp = get_numbers(c[1])
            new_obj.rect = Object()
            new_obj.rect.xpos = tmp[0]
            new_obj.rect.ypos = tmp[1]
            new_obj.rect.width = tmp[2]
            new_obj.rect.height = tmp[3]
        elif key == 'fcn':
            new_obj.fcn = Object()
            name = c[-1]
            if name.startswith('[') and name.endswith(']'):
                name = name[1:-1]
            if name != '' and name != '||':
                logger.info('Translate code from object.')
                new_obj.fcn.code = PARSER.translate_from_project(name, wrapper="GLOBAL")
        elif key == "kind":
            name = int(c[-1])
            if name == 1:
                new_obj.fcn.mode = "once"
            elif name == 0:
                new_obj.fcn.mode = "forever"
        elif key == "text":
            new_obj.text = PROCESS_RTF(content[7:])
            new_obj.plaintext = str(PLAINTEXT).lower()
        elif key == "buffsize":
            new_obj.buffsize = get_numbers(c[-1])
        elif key == "buffer":
            if BUFFERFILES:
                new_obj.buffer = ''
                name = str(uuid.uuid4())
                new_obj.bufferfile = create_files(os.path.join(BASE_DIR, FILESDIR, name + '.bff'), 'wb', content[14:])
            else:
                new_obj.buffer = str(base64.b64encode(content[14:]))[2:-1]
                new_obj.bufferfile = ''
    this_obj.Object.append(new_obj)


def process_object_pro(obj_content, this_obj):
    """Обработка объектов проекта"""
    new_obj = Object()
    new_f = io.BytesIO(obj_content)
    new_f.read(20)
    while new_f.tell() < len(obj_content):
        content = get_key_values(new_f)
        try:
            c = content.decode('utf-16').split('\x00')
        except:
            c = content[:30].decode('utf-16').split('\x00')
        key = c[0]
        if key in {'name', 'type', 'locked?', 'headsync?', 'shown?', 'size', 'pencolor', 'pensize', 'penstate',
                   'heading', 'xpos', 'ypos', 'shapein', 'colorin', 'opacity',
                   'shape', 'turn-angle', 'shortname', 'pathname', 'visible?', 'show-name?'}:
            attr_name = key.replace('?', '').replace('-', '_')
            vars(new_obj)[attr_name] = get_logo_constant(c[1])
        elif key == "have-list":
            have_list = c[-1][4:-1]
            if have_list.strip():
                tree = PARSER.parse(have_list, all_constants=True).children[0]
                tvars = {el.value.source: PARSER.post_order_translation(tree.children[i + 1])
                               for i, el in enumerate(tree.children) if i % 2 == 0}
                for key in tvars:
                    try:
                        tvars[key] = json.loads(tvars[key])
                    except:
                        pass
                if tvars:
                    turtlevars = Object()
                    for tvarname in tvars:
                        vars(turtlevars)[tvarname] = tvars[tvarname]
                    new_obj.turtlevar = turtlevars
        elif key == "rect":
            tmp = get_numbers(c[1])
            new_obj.rect = Object()
            new_obj.rect.xpos = tmp[0]
            new_obj.rect.ypos = tmp[1]
            new_obj.rect.width = tmp[2]
            new_obj.rect.height = tmp[3]
        elif key == "ShapesProcs":
            new_obj.shapesprocs = []
            logger.info('Translate code from ShapesProcs.')
            procs = striprtf(content[38:])
            procsarray = PARSER.make_procsarray(procs)
            process_procedures(new_obj.shapesprocs, procsarray)
        elif key == 'mapshapenumber':
            new_obj.mapshapenumber = c[1][1:-1]
            str123 = new_obj.mapshapenumber.split()[1:]
            new_obj.mapshapenumber = []
            for i in range(0, len(str123), 2):
                tmp_obj = Object()
                tmp_obj.number = int(str123[i])
                tmp_obj.hash = str123[i + 1]
                new_obj.mapshapenumber.append(tmp_obj)
        elif key == 'shapenames':
            new_obj.shapenames = c[1][1:-1]
            str123 = new_obj.shapenames.split()[1:]
            new_obj.shapenames = []
            for i in range(0, len(str123), 2):
                tmp_obj = Object()
                tmp_obj.number = int(str123[i])
                tmp_obj.name = str123[i + 1][1:-1]
                new_obj.shapenames.append(tmp_obj)
        elif key == 'ontimer':
            cur_time, tic, name = c[1][1:-1].split()
            name = name[1:-1]
            new_obj.ontimer = Object()
            new_obj.ontimer.cur = cur_time
            new_obj.ontimer.tic = tic
            if name != '' and name != '||':
                logger.info('Translate code from object pro ontimer.')
                new_obj.ontimer.code = PARSER.translate_from_project(name, wrapper="GLOBAL")
        elif key == "oncolor":
            new_obj.oncolor = []
            name = c[1][1:-1]
            lists = process_lists(name)
            for el in lists:
                col_obj = Object()
                tmp_str = el[1:-1]
                str_list = tmp_str.split()
                col_obj.mode = PARSER.ENV.TRANSLATIONS[str_list[0]] if str_list[0] in PARSER.ENV.TRANSLATIONS else str_list[0]
                tmp_str = ' '.join(str_list[1:])
                if tmp_str != '' and tmp_str != '||':
                    logger.info('Translate code from object pro oncolor. <{}>'.format(tmp_str))
                    col_obj.code = PARSER.translate_from_project(tmp_str, wrapper="GLOBAL")

                new_obj.oncolor.append(col_obj)
        elif key == 'fcn':
            new_obj.fcn = Object()
            name = c[1]
            if name.startswith('[') and name.endswith(']'):
                name = name[1:-1]
            if name != '' and name != '||':
                logger.info('Translate code from object pro fcn.')
                new_obj.fcn.code = PARSER.translate_from_project(name, wrapper="GLOBAL")
        elif key == 'kind':
            name = int(c[-1])
            if name == 1:
                new_obj.fcn.mode = "once"
            elif name == 0:
                new_obj.fcn.mode = "forever"
        elif key in {'ontouching', 'onmessage'}:
            name = c[1][1:-1]
            if name != '' and name != '||':
                logger.info('Translate code from object pro ontouching\onmessage.')
                vars(new_obj)[key] = PARSER.translate_from_project(name, wrapper="GLOBAL")
    this_obj.Object.append(new_obj)


def process_page(page, this_obj, offset=10):
    """Обработка страниц проекта"""
    loc_obj = Object()
    loc_obj.Object = []
    loc_obj.turtle_deamons = []
    loc_obj.deamons_mode = []
    loc_obj.mouse_deamons = []
    new_f = io.BytesIO(page)
    new_f.read(offset)
    while new_f.tell() < len(page):
        content = get_key_values(new_f)
        try:
            c = content.decode('utf-16').split('\x00')
        except:
            c = content[:30].decode('utf-16').split('\x00')
        key = c[0]
        if key in {'name', 'curturtle', 'curtext', 'transition', 'bg', 'bg_alpha'}:
            vars(loc_obj)[key] = get_logo_constant(c[1])
        elif key == "buffer":
            if BUFFERFILES:
                loc_obj.buffer = ''
                name = str(uuid.uuid4())
                loc_obj.bufferfile = create_files(os.path.join(BASE_DIR, FILESDIR, name + '.bff'), 'wb', content[14:])
            else:
                loc_obj.buffer = str(base64.b64encode(content[14:]))[2:-1]
                loc_obj.bufferfile = ''
        elif key == 'freezebg':
            if BUFFERFILES:
                loc_obj.freezebg = ''
                name = str(uuid.uuid4())
                loc_obj.freezebgfile = create_files(os.path.join(BASE_DIR, FILESDIR, name + '.bff'), 'wb', content[18:])
            else:
                loc_obj.freezebg = str(base64.b64encode(content[18:]))[2:-1]
                loc_obj.freezebgfile = ''
        elif key == "object":
            process_object(content, loc_obj)
        elif key == "ObjectPro":
            process_object_pro(content, loc_obj)
        elif key == "turtle-deamons":
            text = content.decode('utf-16')[16:-1]
            # print(PARSER.translate_from_project(text))
            # exit()
            lists = process_lists(text)
            for el in lists:
                tmp_str = el[1:-1]
                if tmp_str:
                    str_list = tmp_str.split()
                    mode = str_list[0]
                    tmp_str = ' '.join(str_list[1:])
                    if tmp_str != '':
                        logger.info('Translate code from turtle-deamons.')
                        code = PARSER.translate_from_project(tmp_str, wrapper="GLOBAL")
                    else:
                        code = ''
                else:
                    mode, code = '', ''
                loc_obj.deamons_mode.append(mode)
                loc_obj.turtle_deamons.append(code)
        elif key == "mouse-deamons":
            text = content.decode('utf-16')[15:-1]
            lists = process_lists(text)
            for el in lists:
                tmp_str = el[1:-1]
                if tmp_str != '':
                    logger.info('Translate code from mouse-deamons.')
                    code = PARSER.translate_from_project(tmp_str, wrapper="GLOBAL")
                else:
                    code = ''
                loc_obj.mouse_deamons.append(code)
    this_obj.pages.append(loc_obj)


def localized_colors():
    colors = {PARSER.ENV.STRINGS['color{}'.format(i)][0] for i in range(16)}
    return {PARSER.ENV.TRANSLATIONS[color]: color for color in colors}


def second_run():
    """Второй проход по содержимому проекта: записываем содержимое в атрибуты объекта, возвращаем полученный объект."""

    global FBASE
    FBASE = open('library.dll', 'rb')  # Библиотека
    f = open(TEMPORARY_FILE, 'rb')

    final_object = Object()
    final_object.compiler_version = __version__
    final_object.project_type = VERSION.lower()
    final_object.compiler_lang = LANGUAGE
    final_object.template_page = PARSER.ENV.STRINGS['template_page'][0]
    final_object.pages = []

    logger.info("Processing project...")

    f.read(32)
    while f.tell() < FILE_LEN:
        content = get_key_values(f)
        c = content[:100].decode('utf-16').split('\x00')
        key = c[0]
        if key == "projectsize":
            # content = [int(i) for i in get_numbers(c[-1])]
            final_object.projectsize = get_logo_constant(c[-1])
        elif key == "procedures":
            logger.info('Translate code from procedures.')
            procs = striprtf(content[34:]).replace('\x00', '')
            procsarray = PARSER.make_procsarray(procs)
            final_object.procedures = []
            process_procedures(final_object.procedures, procsarray)
        elif key == "Bitmaps":
            final_object.Bitmaps = []
            process_bitmaps(content, final_object)
            log_message = "empty" if not final_object.Bitmaps else "loaded"
            logger.info("Bitmaps are {}.".format(log_message))
        elif key == "GlobalShapes":
            process_global_shapes(content, final_object)
        elif key == "page":
            process_page(content, final_object)
        elif key == "templatepage":
            process_page(content, final_object, offset=26)
        elif key == "globals-list":
            globals_list = ' '.join(content.decode('utf-16')[14:-1].split()[1:])
            # print(content.decode('utf-16')[14:-1])
            if globals_list.strip():
                tree = PARSER.parse(globals_list, all_constants=True).children[0]
                # print(tree)
                projectvars = {el.value.source: PARSER.post_order_translation(tree.children[i+1])
                               for i, el in enumerate(tree.children) if i % 2 == 0}
                for key in projectvars:
                    try:
                        projectvars[key] = json.loads(projectvars[key])
                    except:
                        pass
                if projectvars:
                    final_object.globals = projectvars
                # print(projectvars)
    final_object.colorslocalized = localized_colors()
    FBASE.close()
    f.close()
    return final_object


def check_localization_dir_exists(language):
    if language != "English":
        if not os.path.exists(os.path.join(BASE_DIR, language)):
            logger.warning('Localization directory <{}> does not exist. '
                           'Trying to read from <{}>.'.format(os.path.join(BASE_DIR, language),
                                                              os.path.join(BASE_DIR, "English")))
            language = "English"
        else:
            if not os.path.exists(os.path.join(BASE_DIR, "English")):
                logger.error('Localization directory <{}> does not exist. '.format(os.path.join(BASE_DIR, "English")))
                logger.error('Parser cannot proceed.')
                exit()
    if language == "English" and not os.path.exists(os.path.join(BASE_DIR, language)):
        logger.error('Localization directory <{}> does not exist. '.format(os.path.join(BASE_DIR, language)))
        logger.error('Parser cannot proceed.')
        exit()
    return language


if __name__ == "__main__":
    # parse command line arguments
    p = argparse.ArgumentParser()
    p.add_argument('project_file', help="Path to mwx-project.")
    p.add_argument('--bufferfiles', dest='bufferfiles', action='store_true',
                   help="Whether write media to data.js of separate files.")
    p.add_argument('--no-bufferfiles', dest='bufferfiles', action='store_false',
                   help="Whether write media to data.js of separate files.")
    p.set_defaults(bufferfiles=False)
    p.add_argument('--libreoffice', type=str, dest="libreoffice", default="/usr/lib/libreoffice/program/soffice",
                   help="Path to LibreOffice.")
    p.add_argument('--locale', type=str, dest="locale", default="none",
                   help="Language name, e.g. English or Russian. By default the project's language is extracted "
                        "from the   project file itself. This argument can be used to override the project's language.")
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

    try:
        cmd_args = p.parse_args()
    except:
        # для целей тестирования можно запускать этот код без командной строки,
        # тогда нужные аргументы нужно прописать ниже:
        cmd_line = ["test_projects/bug72-asktest.mwx", '--libreoffice',
                    "C:/Program Files (x86)/LibreOffice 5/program/soffice.exe",
                    "--log", "debug"]
        cmd_args = p.parse_args(cmd_line)

    try:
        # устанавливаем уровень логгирования
        LOG_LEVEL = logging.getLevelName(cmd_args.log.upper())
        logging.basicConfig(level=LOG_LEVEL, stream=sys.stdout)
        logging.getLogger('ProjectParser').setLevel(LOG_LEVEL)
        logging.getLogger('LogoParser').setLevel(LOG_LEVEL)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        TEMPORARY_FILE = os.path.join(BASE_DIR, "js-test.dat")
        PROJECT_FILE = cmd_args.project_file
        project_basename = os.path.splitext(os.path.basename(PROJECT_FILE))[0]

        PROCESS_RTF, PLAINTEXT = get_function_for_rtf_processing(cmd_args.libreoffice)

        BUFFERFILES = cmd_args.bufferfiles
        FILESDIR = os.path.join(BASE_DIR, project_basename) if BUFFERFILES else ''
        if FILESDIR:
            if not os.path.exists(FILESDIR):
                os.makedirs(FILESDIR)

        logger.info("Compiler version: {}".format(__version__))
        logger.info("Start reading project.")

        LANGUAGE, VERSION, FILE_LEN = temporary_file_length(PROJECT_FILE, TEMPORARY_FILE)
        if cmd_args.locale != 'none':
            LANGUAGE = cmd_args.locale

        LANGUAGE = check_localization_dir_exists(LANGUAGE)

        object_rules_file = os.path.join(BASE_DIR, 'object-rules.csv')
        TYPES = read_object_rules(object_rules_file)

        PARSER = Parser(locale=LANGUAGE)

        first_run()

        obj = second_run()

        os.remove(TEMPORARY_FILE)

        logger.info("Writing data...")
        with open(os.path.join(BASE_DIR, 'data.js'), 'w', encoding='utf-8') as fo:
            fo.write("projectData = {};".format(obj.to_JSON()))
        logger.info("Processing finished.")
        print('0', file=sys.stderr)
    except Exception as e:
        logger.critical(e)
        print('1', file=sys.stderr)