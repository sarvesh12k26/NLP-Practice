import nltk

#nltk.download()
from nltk.tokenize import sent_tokenize,word_tokenize

text='Hello there, Che Pujara scored a big hundred. Will Australia score 700 to keep the match interesting? I dont think this will happen'
print(sent_tokenize(text))
print(word_tokenize(text))

#Removing stopwords
from nltk.corpus import stopwords
text='Hello there, Che Pujara scored a big hundred. Will Australia score 700 to keep the match interesting? I dont think this will happen'
stop_words=set(stopwords.words('english'))
word_tokens=word_tokenize(text)
filtered_sentence=[w for w in word_tokens if not w in stop_words]
print(word_tokens)
print(filtered_sentence)

#Stemming words
from nltk.stem import PorterStemmer
ps=PorterStemmer()
example=['ride','rides','rider','riding']

for w in example:
    print(ps.stem(w))
 
sentence="When riders are riding their horses, they often think of how cowboy rode horses."
words=word_tokenize(sentence)
for w in words:
    print(ps.stem(w))
    
#Video 2
from nltk.corpus import udhr
print(udhr.raw('English-Latin1'))    

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text=state_union.raw('2005-GWBush.txt')
sample_text=state_union.raw('2006-GWBush.txt')

#Train the PunktTokenizer
custom_sent_tokenizer=PunktSentenceTokenizer(train_text)
tokenized=custom_sent_tokenizer.tokenize(sample_text)

#This function tags each tokenized word with a part of speech
def process_content():
    try:
        for i in tokenized[:5]:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            print(tagged)
            
    except Exception as e:
        print(str(e))

process_content()
#####
#View the nltk tag of speech full-form
#nltk.download()
nltk.help.upenn_tagset()
#####

#Chunking with NLTK
train_text=state_union.raw('2005-GWBush.txt')
sample_text=state_union.raw('2006-GWBush.txt')

custom_sent_tokenizer=PunktSentenceTokenizer(train_text)
tokenized=custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            #combine the part of speech tag with a regular expression
            chunkGram=r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser=nltk.RegexpParser(chunkGram)
            chunked=chunkParser.parse(tagged)
            
            #print the nltk tree
            for subtree in chunked.subtrees(filter=lambda t: t.label()=='Chunk'):
                print(subtree)
            
            #draw the chunks with nltk
            #chunked.draw()
            
    except Exception as e:
        print(str(e))

process_content()

#Chinking with NLTK
train_text=state_union.raw('2005-GWBush.txt')
sample_text=state_union.raw('2006-GWBush.txt')

custom_sent_tokenizer=PunktSentenceTokenizer(train_text)
tokenized=custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            #combine the part of speech tag with a regular expression
            #}{ this means we remove the chink one or more verbs,preposition,determinants,or the word 'to'.
            chunkGram=r"""Chunk: {<.*>+}
                                        }<VB.?|IN|DT|TO>+{"""
            chunkParser=nltk.RegexpParser(chunkGram)
            chunked=chunkParser.parse(tagged)
            
            #print the nltk tree
            for subtree in chunked.subtrees(filter=lambda t: t.label()=='Chunk'):
                print(subtree)
            
            #draw the chunks with nltk
            #chunked.draw()
            
    except Exception as e:
        print(str(e))

process_content()

#Named Entity Recognition
def process_content():
    try:
        for i in tokenized[:5]:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            namedEnt=nltk.ne_chunk(tagged,binary=False)
            
            #draw the chunks with nltk
            namedEnt.draw()
            
    except Exception as e:
        print(str(e))

process_content()


