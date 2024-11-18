import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import InPut

# Output the contents of the list
# print(text_list)

# Split each line of code into segments
def fragment(text_list):
    new_text_list = []
    for line in text_list:
        fragments = [line[i:i+8] for i in range(0, len(line), 8)]
        new_line = ' '.join(fragments)
        new_text_list.append(new_line)
    '''
    for line in text_list:
        fragments = [line[i:i+8] for i in range(len(line)-7)]
        new_line = ' '.join(fragments)
        new_text_list.append(new_line)
    '''
    return new_text_list

# print(new_text_list)

# Convert the list into a DataFrame with a single column
def getText(new_text_list):
    df_train = pd.DataFrame(new_text_list, columns=['Text'])
    return df_train

# text_df = getText()

TaggededDocument = gensim.models.doc2vec.TaggedDocument

# Preprocess the text data
def preprocess_text(text_df):
    tagged_data = []
    for index, row in text_df.iterrows():
        tokens = row['Text'].split()
        tagged_data.append(TaggededDocument(words=tokens, tags=[index]))
    return tagged_data

# Train the Doc2Vec model
def train(c, size=128):
    model = Doc2Vec(c, dm=1, min_count=15, window=5, vector_size=size, sample=0, negative=5, workers=5)
    model.train(c, total_examples=model.corpus_count, epochs=50)
    return model

# Get the vectors
def get_seqvec():
    text_list = InPut.Seq_input()
    new_text_list = fragment(text_list)
    text_df = getText(new_text_list)
    c = preprocess_text(text_df)
    model_dm = train(c)
    vectors = [model_dm.dv[i] for i in range(len(text_df))]
    return vectors

# df.to_excel("E:\data\finally\seqdata\mSequenceVec.xlsx")
