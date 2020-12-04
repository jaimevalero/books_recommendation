"""Main module."""
import books_recommendation.epubtotext as epubtotext
import glob
import os
import pandas as pd
from loguru import logger

logger.info("Start")
ebook_title ='/home/jaimevalero/libros/Mahoma - Cesar Vidal.epub'
ebook_metadata = epubtotext.get_epub_info(ebook_title)
#out = epubtotext.epub_to_clean_text(ebook)   
#print(out)


def get_df_books():
    """
    Iterate trough path and generate a dataframe with the content and metadata for each epub
    """
    BOOKS_PATH='books/*epub'
    ebooks= sorted(glob.glob(BOOKS_PATH) )
    logger.info(ebooks)
    ebooks_array = []
    for ebook in ebooks:
        logger.info(ebook)
        ebook_metadata = epubtotext.get_epub_info(ebook)
        d = { 'text' : epubtotext.epub_to_clean_text(ebook) , **ebook_metadata }
        ebooks_array.append(d)
    df = pd.DataFrame(ebooks_array)   
    return df

df = get_df_books()
df.to_csv("books.csv")


#