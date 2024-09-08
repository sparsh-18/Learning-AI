from langchain.schema import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    db: VectorStore

    def _get_relevant_documents(self, query):
        # calculate embeddings for the query
        emb = self.embeddings.embed_query(query)
        
        # feed embeddings to the chroma to get the relevant documents
        return self.db.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8 # lambda value for tolerance of similar documents
        )

    async def _aget_relevant_documents(self, query):
        return []
