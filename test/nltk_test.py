__author__ = 'Eunhwan Jude Park'
__email__ = 'jude.park.96@navercorp.com'

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

para = "Hello World. It's good to see you. Thanks for buying this book."

print(sent_tokenize(para))
print(word_tokenize(para))


