# Wikipedia discussions extractor

import xml.sax
import subprocess
import mwparserfromhell
import re
import csv

DATA_PATH = "data/frwiki-latest-pages-meta-current1.xml-p1p306134.bz2"
OUTPUT_FILE = "data/frwiki_discussions_categories.csv"
MAX_PAGES = 1000000

class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []

        self.outfile = open(OUTPUT_FILE, "a")
        self.writer = csv.writer(self.outfile)

        self.writer.writerow(["title", "text", "disc", "categories"])
        

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            # Filter to extract only discussions between users
            if "Discussion" in self._values['title']:
                self.parse_page()

    def parse_page(self):

        parsed_disc = self.parse_discussion(self._values['text'])
        categories = self.parse_categories()

        # Flatten ArrayType before writing to CSV file
        # Not using foreign key for title and text create huge duplicates

        for disc in parsed_disc:
            row = (
                self._values['title'], # title 
                self._values['text'],  # raw text
                disc, # parsed discussion sections
                categories # page categories
            )
            self._pages.append(row)

            # Write to output file
            self.writer.writerow(row)

    def parse_categories(self):

        categories = re.compile(r'{{(.*?)}}', re.DOTALL).search(self._values['text'])
        if categories:
            categories = categories.group().strip()
            flatten = categories[2:-2].replace("\n", "").split("|")

            # If regular discussion
            if flatten[0].strip() == "Wikiprojet": # filter noise
                categories = [flatten[i] for i in range(1, len(flatten) - 1, 2)]
            
            # If page translated
            elif "Traduit" in flatten[0]:
                categories = flatten[-1]
            
            # Else, no labeled data
            else:
                categories = None

        return categories


    def parse_discussion(self, page):
        wiki = mwparserfromhell.parse(page)
        text = wiki.strip_code().strip()
        sections = wiki.get_sections()

        pattern = re.compile(r"== *[a-zA-Z].* *==")

        parsed_sections = pattern.split(sections[0].strip_code().strip())

        return parsed_sections

def main():

    # Object for handling xml
    handler = WikiXmlHandler()
    
    # Parsing object
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    
    # Iteratively process file
    for line in subprocess.Popen(['bzcat'], 
                                stdin = open(DATA_PATH), 
                                stdout = subprocess.PIPE).stdout:
        parser.feed(line)

        if max_n:
            if len(handler._pages) >= MAX_PAGES:
                break
    
    handler.outfile.close()

if __name__ == "__main__":
    main()

