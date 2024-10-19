from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings - deprecated
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma - deprecated
from langchain_chroma import Chroma

CHROMA_DIR = 'facts-chat-chroma'

load_dotenv()

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader('facts.txt')
docs = loader.load_and_split(text_splitter)
# print(docs)

huggin_face_embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-l6-v2' # It maps sentences & paragraphs to a 384 dimensional dense vector space
)

# the function takes a doc from list of docs, embed them and store them in the directory in sqlite format
db = Chroma.from_documents(
    documents=docs,
    embedding=huggin_face_embeddings,
    persist_directory=CHROMA_DIR
)

result = db.similarity_search_with_score(
    'What is an interesting fact about the English Language?'
)

print('Results:\n')

# res is of type Tuple[Langchain Document, float]
for res in result:
    print(res[0].page_content, end='\n')


# with no value in k
'''
Results:

4. A snail can sleep for three years.
5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'
6. The elephant is the only mammal that can't jump.
76. The word "OK" stands for "oll korrect," a deliberate misspelling of "all correct."
77. The only letter that doesnâ€™t appear on the Periodic Table is J.
78. Sheep donâ€™t drink from running water.
1. "Dreamt" is the only English word that ends with the letters "mt."
2. An ostrich's eye is bigger than its brain.
3. Honey is the only natural food that is made without destroying any kind of life.
96. Some animals have blue, green, or violet blood.
97. A strawberry isn't a berry but a banana is.
98. A Swiss passport is the world's most accepted passport.
'''

# ================ notes ================
# Embeddings are a way to represent words in a way that a machine learning model can understand.
# The embeddings are vectors of numbers that represent the meaning of a word.
# Embeddings lie from -1 to 1 that score on how much a word is related to another word.
# Dimensions are the number of numbers in the vector that represents the word.
# Similarity is found using cosine similarity or squared l2 distance.
# squared l2 distance is the sum of the squared differences between the two vectors.
# cosine similarity is the cosine of the angle between the two vectors.

# first the text is chunked into smaller pieces then separated by the separator.
# if chunk size is smaller than the next separator then the chunk is increased to include the separator.
# chunk overlap is the number of characters that are included in the next chunk from the previous chunk.