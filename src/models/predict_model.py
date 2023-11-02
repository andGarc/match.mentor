from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# iInitialize the model
model = SentenceTransformer('bert-base-nli-mean-tokens')

'''
BERT Model. Uses Cosine similarity.
Function calculates the similarity between a mentor a the mentees.
Params:
    sentences (list): list of sentence where the sentence at 0 is the mentor
                        and the rest are mentees
Returns:
    result (nparray): array with the cosine similarity between the mentor and
                        mentees
'''
def bert(sentences):
    # enconde sentences
    sentence_embeddings = model.encode(sentences)

    # find most similar sentence between mentor [0] and mentees [1:]
    # let's calculate cosine similarity
    results = cosine_similarity(
        [sentence_embeddings[0]],
        sentence_embeddings[1:]
    )

    results = results.tolist()[0]

    return results