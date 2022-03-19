#!/usr/bin/env python

import argparse
import collections
import itertools
import logging
import os
import sys
import unicodedata
from abc import abstractmethod

import d64
import more_itertools
import numpy as np
import regex
from lxml import etree

CORPUS_TRIGRAMS = collections.defaultdict(lambda: 0, [(x.split('\t')[0], int(x.split('\t')[1].strip())) for x in open(
    os.path.join(os.path.dirname(__file__), 'trigram_counts.tsv'))])

CUS_MAPPING = {}
for l in open('pet_unicode.txt'):
    if l.startswith('#'): continue
    k, v = tuple(l.strip().split(';')[:2])
    CUS_MAPPING[int(k, 16)] = v + ' (CUS)'


def get_unicode_str(char):
    try:
        return f'U+{ord(char):X} ' + unicodedata.name(char)
    except ValueError:
        return f'U+{ord(char):X} ' + CUS_MAPPING[ord(char)]


def ischar_petscii(i):
    if i in [10, 11, 141]: return True
    if 32 <= i <= 122: return True
    if 193 <= i <= 218: return True
    return False


def ischar_ascii(i):
    if i in [10, 13, 11]: return True
    if 32 <= i <= 126: return True
    return False


def ischar(i):
    if 65 <= i <= 90: return True
    if 97 <= i <= 122: return True
    return False


def decode_petscii(c):
    # return c.decode('petscii_c64en_lc', errors='replace').replace('\r', '\n')
    x = c.replace(b'\n\r', b'\r').replace(b'\r\n', b'\r').replace(b'\n', b'\r').decode('petscii_c64en_lc',
                                                                                       errors='replace')
    x = x.replace('\r', '\n')
    return x


def decode_ascii(c):
    x = c.replace(b'\n\r', b'\n').replace(b'\r\n', b'\n').replace(b'\r', b'\n').decode('ascii', errors='replace')
    x = regex.sub(r'[\x00-\x08\x0b-\x1f]', '\N{REPLACEMENT CHARACTER}', x)
    return x


def entropy(content):
    count = collections.Counter(content)
    probs = [c / len(content) for c in count.values()]
    return -sum([p * np.log(p) / np.log(2.0) for p in probs])


def read_diskmag(diskmag_file):
    with d64.DiskImage(diskmag_file) as disk:
        for p in disk.glob(b'*'):
            if b'=' in p.name: continue
            name = p.name.decode(encoding='petscii_c64en_lc', errors='replace')

            try:
                f = disk.path(p.name).open()
                content = f.read()
            except (ValueError, AttributeError) as e:
                logging.info(f'Konnte Datei {name} in {diskmag_file} nicht lesen. Überspringe. {e}')
                continue

            if len(content) > f.entry.size * 256:
                logging.info(
                    f'Datei {name} in {diskmag_file} ist {np.ceil(len(content) / 256).astype(int)} blöcke lang, Verzeichnis sagt aber {f.entry.size}. Überspringe.')
                continue

            yield name, content


def classify_content(content):
    if entropy(content) > 7:
        return 'compressed'
    max_sma = np.max(np.convolve(np.array([ischar(x) for x in content]).astype(int), np.ones(20) / 20, mode='valid'))
    if max_sma < 0.5:
        return 'code'
    ascii_chars = np.array([ischar_ascii(x) for x in content]).astype(int).mean()
    petscii_chars = np.array([ischar_petscii(x) for x in content]).astype(int).mean()
    if petscii_chars > 0.95 or ascii_chars > 0.95:
        if petscii_chars > ascii_chars:
            return 'petscii'
        else:
            return 'ascii'

    return 'unknown'


def insert_newlines(c):
    if regex.match(r'^.?\N{REPLACEMENT CHARACTER}', c):
        # das ist ziemlich sicher eine PRG Load Adresse
        c = c[2:]

    scores = {}
    for col_len in range(40, 81):
        rows = [''.join(x) for x in more_itertools.chunked(c, n=col_len)]
        score = 0
        if len(rows[-1]) != col_len:
            score = score - 10
        for r in rows:
            if regex.search(r'^[ \-.,]', r):
                score = score - 1
            if regex.search(r'^\p{L}[ .,;]', r):
                score = score - 1
            if regex.search(r' \p{L}$', r):
                score = score - 1
            if regex.search(r'[\-,.]$', r):
                score = score + 1
            if regex.search(r'\p{Ll}\p{Lu}', r):
                score = score - 1
        scores[col_len] = score

    best_col_len, score = max(scores.items(), key=lambda l: l[1])
    rows = [''.join(x) for x in more_itertools.chunked(c, n=best_col_len)]
    return '\n'.join(rows), best_col_len


def fix_umlaute(c):
    matches = list(regex.finditer(
        r'[a-zA-Z]([^\p{Latin} \p{N}\r\n\t\p{Sc}\p{Sm}\N{CHECK MARK}@.,?!;/(){}\':\-|"\uf100-\uf10f\ufffd])[a-zA-Z ]',
        c))
    if len(matches) > 0:
        replacement_chars = set()
        trigrams = list()
        for m in matches:
            replacement_chars.add(m.group(1))
            trigrams.append(c[m.start(1) - 1:m.end(1) + 1])

        if len(replacement_chars) > len('äöüÄÖÜß'):
            raise ValueError(f'Mehr unbekannte Zeichen als Umlaute vorhanden.')

        mapping_scores = {}
        for mapping in itertools.permutations(list('äöüÄÖÜß'), r=len(replacement_chars)):
            trans = dict(zip(replacement_chars, mapping))
            score = 0
            translated_trigrams = [t.translate(str.maketrans(trans)) for t in trigrams]
            for t in translated_trigrams:
                prob = CORPUS_TRIGRAMS[t] / CORPUS_TRIGRAMS[t[1]]
                if prob == 0:
                    score += -50
                else:
                    score += np.log(prob)

            mapping_scores[mapping] = score

        best_mapping, best_score = max(mapping_scores.items(), key=lambda l: l[1])
        if best_score < -50 * len(trigrams) * 2 / 3:
            raise ValueError(f'Keine geeignete Belegung für Umlaute gefunden.')

        trans = dict(zip(replacement_chars, best_mapping))
        substitutions = [(i, orig, trans[orig]) for i, orig in enumerate(c) if orig in trans.keys()]
        c = c.translate(str.maketrans(trans))

        return c, substitutions
    else:
        return c, {}


class OutputWriter:

    @abstractmethod
    def begin(self):
        return

    @abstractmethod
    def begin_diskmag(self, name):
        return

    @abstractmethod
    def write_text_file(self, filename, content, linebreaks_added=False, substitutions=None):
        return

    @abstractmethod
    def write_binary_file(self, filename, content):
        return

    @abstractmethod
    def end_diskmag(self):
        return

    @abstractmethod
    def end(self):
        return


class TEIWriter(OutputWriter):

    def __init__(self):
        self.xf = None
        self.tei_root = None
        self.text_root = None
        self.disk_root = None

    def begin(self):
        self.xf = etree.xmlfile(sys.stdout.buffer, buffered=False, encoding='utf-8').__enter__()
        self.xf.write_declaration(standalone=True)
        # xf.write_doctype('<!DOCTYPE root SYSTEM "some.dtd">')

        self.tei_root = self.xf.element('TEI', xmlns='http://www.tei-c.org/ns/1.0')
        self.tei_root.__enter__()
        self.xf.write('\n  ')
        with self.xf.element('teiHeader'):
            pass

        self.xf.write('\n  ')
        self.text_root = self.xf.element('text')
        self.text_root.__enter__()
        self.xf.write('\n')

    def begin_diskmag(self, name):
        self.xf.write('    ')
        self.disk_root = self.xf.element('div1', type='diskmag', name=name)
        self.disk_root.__enter__()
        self.xf.write('\n')

    def write_text_file(self, filename, content, linebreaks_added=False, substitutions=None):
        formatted_content = []
        for i, char in enumerate(content):
            if substitutions is None:
                formatted_content.append(char)
            else:
                try:
                    _, orig, replace = next(x for x in substitutions if x[0] == i)
                    subst = etree.Element('subst')
                    etree.SubElement(subst, 'del').text = orig
                    etree.SubElement(subst, 'add').text = replace
                    formatted_content.append(subst)
                except StopIteration:
                    formatted_content.append(char)

        self.xf.write('      ')
        with self.xf.element('div2', name=filename, type='file'):
            with self.xf.element('p'):
                self.xf.write('\n')
                lineno = 1
                for l in more_itertools.split_at(formatted_content, lambda x: x == '\n'):
                    self.xf.write('    ')
                    if linebreaks_added:
                        with self.xf.element('supplied', reason='omitted-in-original'):
                            self.xf.write(etree.Element('lb', n=str(lineno)))
                        self.xf.write(*l)
                    else:
                        self.xf.write(etree.Element('lb', n=str(lineno)), *l)
                    self.xf.write('\n')
                    lineno = lineno + 1

    def write_binary_file(self, filename, content):
        self.xf.write('      ')
        self.xf.write(etree.Element('div2', name=filename, type='file', content='binary', filesize=str(len(content))))
        self.xf.write('\n')

    def end_diskmag(self):
        self.xf.write('    ')
        self.disk_root.__exit__(None, None, None)
        self.xf.write('\n')

    def end(self):
        self.text_root.__exit__(None, None, None)
        self.tei_root.__exit__(None, None, None)


class HTMLWriter(OutputWriter):

    def __init__(self):
        self.xf = None
        self.html_root = None
        self.body_root = None

    def begin(self):
        self.xf = etree.htmlfile(sys.stdout.buffer, buffered=False, encoding='utf-8').__enter__()
        self.xf.write_doctype('<!DOCTYPE HTML>')

        self.html_root = self.xf.element('html')
        self.html_root.__enter__()
        self.xf.write('\n  ')
        with self.xf.element('head'):
            self.xf.write(etree.fromstring(
                '<style>pre {max-width: 1000px; overflow-wrap: break-word; white-space: pre-wrap}</style>'))

        self.xf.write('\n  ')
        self.body_root = self.xf.element('body')
        self.body_root.__enter__()
        self.xf.write('\n')

    def begin_diskmag(self, name):
        self.xf.write('    ')
        with self.xf.element('h2'):
            self.xf.write(name)
        self.xf.write('\n')

    def write_text_file(self, filename, content, linebreaks_added=False, substitutions=None):
        self.xf.write('     ')
        with self.xf.element('h3'):
            self.xf.write(filename)
        self.xf.write('\n     ')
        with self.xf.element('pre'):
            self.xf.write(content)
        self.xf.write('\n')

    def write_binary_file(self, filename, content):
        pass

    def end_diskmag(self):
        self.xf.write('\n')

    def end(self):
        self.body_root.__exit__(None, None, None)
        self.html_root.__exit__(None, None, None)


class StdoutWriter(OutputWriter):

    def begin(self):
        pass

    def begin_diskmag(self, name):
        pass

    def write_text_file(self, filename, content, linebreaks_added=False, substitutions=None):
        print(content)

    def write_binary_file(self, filename, content):
        pass

    def end_diskmag(self):
        pass

    def end(self):
        pass


class FilesWriter(OutputWriter):

    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.current_diskmag = None

    def begin(self):
        pass

    def begin_diskmag(self, name):
        self.current_diskmag = name

    def write_text_file(self, filename, content, linebreaks_added=False, substitutions=None):
        filename = regex.sub(r'[^\w ]', '-', filename)
        with open(os.path.join(self.output_directory, self.current_diskmag + '_' + filename), 'w') as f:
            print(content, file=f)

    def write_binary_file(self, filename, content):
        pass

    def end_diskmag(self):
        pass

    def end(self):
        pass


def main():
    parser = argparse.ArgumentParser(description='Lese D64-Diskmags und extrahiere Plaintext-Daten.')
    parser.add_argument('--fix-umlaute', default=True,
                        help='Ersetze für jede Datei unbelegte Codepoints mit geeigneten Umlauten mittels einer automatisch generierten Zuordnung',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--fix-umbrueche', default=True,
                        help='Füge für jede Datei regelmäßige Zeilenumbrüche ein, falls diese fehlen',
                        action=argparse.BooleanOptionalAction)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--xml',
                       help='Schreibe Ergebnisse als einzelne pseudo-TEI-codierte XML-Datei in die Standardausgabe',
                       action='store_const', dest='outtype', const='tei')
    group.add_argument('--html', help='Schreibe Ergebnisse als einzelne HTML-Datei in die Standardausgabe',
                       action='store_const', dest='outtype', const='html')
    group.add_argument('--stdout', default=True,
                       help='Schreibe Ergebnisse unformatiert und konkateniert in die Standardausgabe',
                       action='store_const', dest='outtype', const='stdout')
    group.add_argument('--writefiles', metavar='VERZEICHNIS', nargs='?',
                       help='Lege für jedes Ergebnis eine Datei in VERZEICHNIS an', const='.', default=None)
    parser.add_argument('-v', '--verbose', help="Gebe ausführliche Informationen aus", action="store_const",
                        dest="loglevel", const=logging.INFO)
    parser.add_argument('diskmags', metavar='FILE', type=str, nargs='+', help='Eine oder mehrere D64-Diskmag-Abbilder')
    parser.set_defaults(outtype='stdout')
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    if args.outtype == 'tei':
        writer = TEIWriter()
    elif args.outtype == 'html':
        writer = HTMLWriter()
    elif args.writefiles is not None:
        writer = FilesWriter(args.writefiles)
    else:
        assert args.outtype == 'stdout'
        writer = StdoutWriter()

    writer.begin()

    for diskmag_file in args.diskmags:
        writer.begin_diskmag(os.path.basename(diskmag_file))

        for name, content in read_diskmag(diskmag_file):
            name = regex.sub(r'[^a-zA-Z0-9_\-.]', '\N{REPLACEMENT CHARACTER}', name)
            logging.info(f'Bearbeite Datei {name} in {diskmag_file}')
            classification = classify_content(content)
            if classification == 'petscii':
                c = decode_petscii(content)
            elif classification == 'ascii':
                c = decode_ascii(content)
            else:
                logging.info(f'Datei {name} in {diskmag_file} wurde nicht als Text klassifiziert. Überspringe.')
                writer.write_binary_file(name, content)
                continue
            c = c.replace('\N{NO-BREAK SPACE}', ' ')

            substitutions = None
            if args.fix_umlaute:
                try:
                    c, substitutions = fix_umlaute(c)
                    for k, v in set((s[1], s[2]) for s in substitutions):
                        logging.info(
                            f'In Datei {name} in {diskmag_file}: ersetze {get_unicode_str(k)} mit {get_unicode_str(v)}.')
                except ValueError as e:
                    logging.warning(
                        f'In Datei {name} in {diskmag_file}: Konnte keine geeignete Substitution für Umlaute ermitteln. {e}')

            best_col_len = None
            if args.fix_umbrueche and '\n' not in c and 50 < len(c) < 5000:
                c, best_col_len = insert_newlines(c)
                logging.info(
                    f'In Datei {name} in {diskmag_file}: füge alle {best_col_len} Zeichen einen Zeilenumbruch hinzu.')

            num_replacement_chars = len(regex.findall(r'\N{REPLACEMENT CHARACTER}', c))
            if num_replacement_chars > 0:
                logging.warning(
                    f'In Datei {name} in {diskmag_file}: ersetze {num_replacement_chars} Bytes mit Ersatzzeichen U+FFFD REPLACEMENT CHARACTER.')

            writer.write_text_file(name, c, linebreaks_added=best_col_len is not None, substitutions=substitutions)

        writer.end_diskmag()

    writer.end()


if __name__ == "__main__":
    main()
