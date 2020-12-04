
import zipfile
from lxml import etree

def get_epub_info(fname):
    ns = {
        'n':'urn:oasis:names:tc:opendocument:xmlns:container',
        'pkg':'http://www.idpf.org/2007/opf',
        'dc':'http://purl.org/dc/elements/1.1/'
    }

    # prepare to read from the .epub file
    zip = zipfile.ZipFile(fname)

    # find the contents metafile
    txt = zip.read('META-INF/container.xml')
    tree = etree.fromstring(txt)
    cfname = tree.xpath('n:rootfiles/n:rootfile/@full-path',namespaces=ns)[0]

    # grab the metadata block from the contents metafile
    cf = zip.read(cfname)
    tree = etree.fromstring(cf)
    p = tree.xpath('/pkg:package/pkg:metadata',namespaces=ns)[0]

    # repackage the data
    res = {}
    for s in ['title','language','creator','date','identifier']:
        res[s] = p.xpath('dc:%s/text()'%(s),namespaces=ns)[0]

    return res


import ebooklib
from ebooklib import epub
def epub2thtml(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapters.append(item.get_content())
    return chapters
#epub2thtml(ebook)    

from bs4 import BeautifulSoup
blacklist = [   '[document]',   'noscript', 'header',   'html', 'meta', 'head','input', 'script',  "style" ]
# there may be more elements you don't want, such as "style", etc.
def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    text = soup.find_all(text=True)
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output

def thtml2ttext(thtml):
    Output = []
    for html in thtml:
        text =  chap2text(html)
        Output.append(text)
    return Output

def epub2text(epub_path):
    chapters = epub2thtml(epub_path)
    ttext = thtml2ttext(chapters)
    return ttext

def filter_nonprintable(text):
    import itertools
    # Use characters of control category
    nonprintable = itertools.chain(range(0x00,0x20),range(0x7f,0xa0))
    # Use translate to remove all non-printable characters
    return text.translate({character:None for character in nonprintable})
def epub_to_clean_text(ebook):
    """
    Convert an epub to text
    Arguments:
    ----------
    ebook: Path to the file
    """
    out = epub2text(ebook)
    text=""
    for chapter in out : 
        text += filter_nonprintable(chapter)    
    return text


