
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess documents
loader = PyPDFDirectoryLoader("/content/drive/MyDrive/")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Set up embeddings and vector store
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
vectorestore = Chroma.from_documents(chunks, embeddings)
retriever = vectorestore.as_retriever(search_kwargs={"k": 5})

# Initialize LlamaCpp model
llm = LlamaCpp(
    model_path="/content/drive/MyDrive/BioMistral-7B.Q4_K_M.gguf",
    temperature=0.2,
    max_tokens=2000,
    top_p=1
)

# Set up prompt template
template = """
<|context|>
You are Medical Assistant that follows the instructions and generate the accurate response based on the query and the context provided.
Please be truthful and give direct answers.
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# API Endpoint
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        response = rag_chain.invoke(query)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
