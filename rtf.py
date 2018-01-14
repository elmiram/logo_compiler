# -*- coding: utf-8 -*-

"""
Extract text in RTF Files. Refactored to use with Python 3.x
Source:
    http://stackoverflow.com/a/188877
Code created by Markus Jarderot: http://mizardx.blogspot.com
"""

import logging
import platform
import os
import re
import uuid

from subprocess import Popen, PIPE


def striprtf(text):
    pattern = re.compile(r"\\([a-z]{1,32})(-?\d{1,10})?[ ]?|\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|(.)", re.I)
    # control words which specify a "destionation".
    destinations = frozenset((
        'aftncn', 'aftnsep', 'aftnsepc', 'annotation', 'atnauthor', 'atndate', 'atnicn', 'atnid',
        'atnparent', 'atnref', 'atntime', 'atrfend', 'atrfstart', 'author', 'background',
        'bkmkend', 'bkmkstart', 'blipuid', 'buptim', 'category', 'colorschememapping',
        'colortbl', 'comment', 'company', 'creatim', 'datafield', 'datastore', 'defchp', 'defpap',
        'do', 'doccomm', 'docvar', 'dptxbxtext', 'ebcend', 'ebcstart', 'factoidname', 'falt',
        'fchars', 'ffdeftext', 'ffentrymcr', 'ffexitmcr', 'ffformat', 'ffhelptext', 'ffl',
        'ffname', 'ffstattext', 'field', 'file', 'filetbl', 'fldinst', 'fldrslt', 'fldtype',
        'fname', 'fontemb', 'fontfile', 'fonttbl', 'footer', 'footerf', 'footerl', 'footerr',
        'footnote', 'formfield', 'ftncn', 'ftnsep', 'ftnsepc', 'g', 'generator', 'gridtbl',
        'header', 'headerf', 'headerl', 'headerr', 'hl', 'hlfr', 'hlinkbase', 'hlloc', 'hlsrc',
        'hsv', 'htmltag', 'info', 'keycode', 'keywords', 'latentstyles', 'lchars', 'levelnumbers',
        'leveltext', 'lfolevel', 'linkval', 'list', 'listlevel', 'listname', 'listoverride',
        'listoverridetable', 'listpicture', 'liststylename', 'listtable', 'listtext',
        'lsdlockedexcept', 'macc', 'maccPr', 'mailmerge', 'maln', 'malnScr', 'manager', 'margPr',
        'mbar', 'mbarPr', 'mbaseJc', 'mbegChr', 'mborderBox', 'mborderBoxPr', 'mbox', 'mboxPr',
        'mchr', 'mcount', 'mctrlPr', 'md', 'mdeg', 'mdegHide', 'mden', 'mdiff', 'mdPr', 'me',
        'mendChr', 'meqArr', 'meqArrPr', 'mf', 'mfName', 'mfPr', 'mfunc', 'mfuncPr', 'mgroupChr',
        'mgroupChrPr', 'mgrow', 'mhideBot', 'mhideLeft', 'mhideRight', 'mhideTop', 'mhtmltag',
        'mlim', 'mlimloc', 'mlimlow', 'mlimlowPr', 'mlimupp', 'mlimuppPr', 'mm', 'mmaddfieldname',
        'mmath', 'mmathPict', 'mmathPr', 'mmaxdist', 'mmc', 'mmcJc', 'mmconnectstr',
        'mmconnectstrdata', 'mmcPr', 'mmcs', 'mmdatasource', 'mmheadersource', 'mmmailsubject',
        'mmodso', 'mmodsofilter', 'mmodsofldmpdata', 'mmodsomappedname', 'mmodsoname',
        'mmodsorecipdata', 'mmodsosort', 'mmodsosrc', 'mmodsotable', 'mmodsoudl',
        'mmodsoudldata', 'mmodsouniquetag', 'mmPr', 'mmquery', 'mmr', 'mnary', 'mnaryPr',
        'mnoBreak', 'mnum', 'mobjDist', 'moMath', 'moMathPara', 'moMathParaPr', 'mopEmu',
        'mphant', 'mphantPr', 'mplcHide', 'mpos', 'mr', 'mrad', 'mradPr', 'mrPr', 'msepChr',
        'mshow', 'mshp', 'msPre', 'msPrePr', 'msSub', 'msSubPr', 'msSubSup', 'msSubSupPr', 'msSup',
        'msSupPr', 'mstrikeBLTR', 'mstrikeH', 'mstrikeTLBR', 'mstrikeV', 'msub', 'msubHide',
        'msup', 'msupHide', 'mtransp', 'mtype', 'mvertJc', 'mvfmf', 'mvfml', 'mvtof', 'mvtol',
        'mzeroAsc', 'mzeroDesc', 'mzeroWid', 'nesttableprops', 'nextfile', 'nonesttables',
        'objalias', 'objclass', 'objdata', 'object', 'objname', 'objsect', 'objtime', 'oldcprops',
        'oldpprops', 'oldsprops', 'oldtprops', 'oleclsid', 'operator', 'panose', 'password',
        'passwordhash', 'pgp', 'pgptbl', 'picprop', 'pict', 'pn', 'pnseclvl', 'pntext', 'pntxta',
        'pntxtb', 'printim', 'private', 'propname', 'protend', 'protstart', 'protusertbl', 'pxe',
        'result', 'revtbl', 'revtim', 'rsidtbl', 'rxe', 'shp', 'shpgrp', 'shpinst',
        'shppict', 'shprslt', 'shptxt', 'sn', 'sp', 'staticval', 'stylesheet', 'subject', 'sv',
        'svb', 'tc', 'template', 'themedata', 'title', 'txe', 'ud', 'upr', 'userprops',
        'wgrffmtfilter', 'windowcaption', 'writereservation', 'writereservhash', 'xe', 'xform',
        'xmlattrname', 'xmlattrvalue', 'xmlclose', 'xmlname', 'xmlnstbl',
        'xmlopen',
    ))
    # Translation of some special characters.
    specialchars = {
        'par': '\n',
        'sect': '\n\n',
        'page': '\n\n',
        'line': '\n',
        'tab': '\t',
        'emdash': '\u2014',
        'endash': '\u2013',
        'emspace': '\u2003',
        'enspace': '\u2002',
        'qmspace': '\u2005',
        'bullet': '\u2022',
        'lquote': '\u2018',
        'rquote': '\u2019',
        'ldblquote': '\201C',
        'rdblquote': '\u201D',
    }
    stack = []
    ignorable = False  # Whether this group (and all inside it) are "ignorable".
    ucskip = 1  # Number of ASCII characters to skip after a unicode character.
    curskip = 0  # Number of ASCII characters left to skip
    out = []  # Output buffer.
    if not isinstance(text, str):
        try:
            text = text.decode()
        except:
            text = text.decode('utf-16')
    # print(text)
    for match in pattern.finditer(text):
        word, arg, hex, char, brace, tchar = match.groups()
        if brace:
            curskip = 0
            if brace == '{':
                # Push state
                stack.append((ucskip, ignorable))
            elif brace == '}':
                # Pop state
                ucskip, ignorable = stack.pop()
        elif char:  # \x (not a letter)
            curskip = 0
            if char == '~':
                if not ignorable:
                    out.append('\xA0')
            elif char in '{}\\':
                if not ignorable:
                    out.append(char)
            elif char == '*':
                ignorable = True
        elif word:  # \foo
            curskip = 0
            if word in destinations:
                ignorable = True
            elif ignorable:
                pass
            elif word in specialchars:
                out.append(specialchars[word])
            elif word == 'uc':
                ucskip = int(arg)
            elif word == 'u':
                c = int(arg)
                if c < 0: c += 0x10000
                if c > 127:
                    out.append(chr(c))  # NOQA
                else:
                    out.append(chr(c))
                curskip = ucskip
        elif hex:  # \'xx
            # print(hex)
            if curskip > 0:
                curskip -= 1
            elif not ignorable:
                c = int(hex, 16)
                if c > 127 and platform.system() == "Windows":
                    out.append(chr(c+848))  # NOQA
                else:
                    out.append(chr(c))
        elif tchar:
            if curskip > 0:
                curskip -= 1
            elif not ignorable:
                out.append(tchar)
    return ''.join(out).replace('\u0000', '')


def rtf2html(libreoffice_path, text):
    if not isinstance(text, str):
        try:
            text = text.decode()
        except:
            text = text.decode('utf-16')
    text = text.replace('\x00', '')

    fname = str(uuid.uuid4())
    with open('{}.rtf'.format(fname), 'w', encoding='utf-8') as f:
        f.write(text)

    p = Popen([libreoffice_path,
               '--headless',
               '--convert-to', 'html', '{}.rtf'.format(fname)],
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p.wait()  # Wait for child process to terminate.
    # o, e = p.communicate(b"input data that is passed to subprocess' stdin")
    logging.debug('Successfully converted RTF to HTML with LibreOffice.')

    with open('{}.html'.format(fname), 'r') as f:
        text = f.read()

    os.remove('{}.rtf'.format(fname))
    os.remove('{}.html'.format(fname))

    return text


def libreoffice_exists(path):
    # EXTENSION = '.exe' if platform.system() == "Windows" else ''
    # LIBREOFFICE_PATH = '{PATH}/program/soffice{EXTENSION}'.format(PATH=path, EXTENSION=EXTENSION)
    # if os.path.exists(path) and os.path.exists(LIBREOFFICE_PATH):
    if os.path.exists(path):
        return path  # LIBREOFFICE_PATH
    return ''


def get_function_for_rtf_processing(path):
    LIBREOFFICE_PATH = libreoffice_exists(path)
    if LIBREOFFICE_PATH:
        def process_rtf(text):
            libreoffice_path = LIBREOFFICE_PATH
            t = rtf2html(libreoffice_path, text)
            return t
        return process_rtf, False
    return striprtf, True