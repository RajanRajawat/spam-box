# pip install pymongo sentence-transformers

#Mongo Connections (.env)
from pymongo import MongoClient

MONGO_URI = "fix it"

client = MongoClient(MONGO_URI)
db = client["API_task"]
collection = db["embeddings"]

 
#Embedding Creation
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

texts = [
    "WerqLabs is a technology company focused on AI-driven digital transformation.",
    "WerqLabs provides scalable and secure software solutions for enterprises.",
    "WerqLabs works across multiple industries including healthcare and finance.",
    "WerqLabs delivers solutions that align business processes with technology.",
    "WerqLabs offers services in software development and cloud computing.",
    "WerqLabs specializes in AI and machine learning based solutions.",
    "WerqLabs helps businesses streamline processes through automation.",
    "WerqLabs builds intelligent systems that adapt to business needs.",
    "WerqLabs has expertise in multiple technology stacks and frameworks.",
    "WerqLabs provides API and SDK integration services.",
    
    "WerqLabs develops web and mobile applications for clients.",
    "WerqLabs provides chatbot development and automation solutions.",
    "WerqLabs works on data-driven services and analytics platforms.",
    "WerqLabs offers workflow management solutions for enterprises.",
    "WerqLabs builds scalable backend systems and APIs.",
    "WerqLabs integrates third-party services like payment and messaging APIs.",
    "WerqLabs develops enterprise-grade applications using modern technologies.",
    "WerqLabs uses cloud platforms like AWS and Azure.",
    "WerqLabs builds applications using Python, Node.js, and React.",
    "WerqLabs supports digital transformation for global businesses.",
    
    "WerqLabs offers internship opportunities in software development.",
    "WerqLabs internships focus on real-world project experience.",
    "Interns at WerqLabs work on backend and frontend systems.",
    "WerqLabs provides on-the-job training for interns.",
    "WerqLabs encourages continuous learning and skill development.",
    "Interns at WerqLabs collaborate with developers and designers.",
    "WerqLabs internship programs include testing and QA roles.",
    "WerqLabs internships require knowledge of programming basics.",
    "WerqLabs interns work on debugging and improving applications.",
    "WerqLabs internships include exposure to Agile methodologies.",
    
    "WerqLabs offers roles in QA testing and software development.",
    "QA interns at WerqLabs design and execute test cases.",
    "QA interns identify and track software bugs and issues.",
    "Developers at WerqLabs build scalable and efficient systems.",
    "WerqLabs developers work with technologies like .NET and C#.",
    "WerqLabs emphasizes writing clean and maintainable code.",
    "WerqLabs developers collaborate with cross-functional teams.",
    "WerqLabs ensures performance and reliability in applications.",
    "WerqLabs teams follow structured software development life cycles.",
    "WerqLabs encourages problem-solving and analytical thinking.",
    
    "WerqLabs provides staff augmentation services for businesses.",
    "WerqLabs helps companies scale teams quickly with skilled developers.",
    "WerqLabs offers flexible hiring and workforce solutions.",
    "WerqLabs provides cost-effective IT staffing solutions.",
    "WerqLabs helps businesses reduce recruitment and training costs.",
    "WerqLabs offers experienced professionals for critical projects.",
    "WerqLabs ensures high-quality IT infrastructure for clients.",
    "WerqLabs supports businesses with scalable team expansion.",
    
    "WerqLabs provides UI and UX design services.",
    "WerqLabs creates responsive and user-friendly interfaces.",
    "WerqLabs designs branding and digital experiences.",
    "WerqLabs offers social media content creation services.",
    "WerqLabs helps improve brand awareness through digital content.",
    "WerqLabs produces animation and visual content for businesses.",
    "WerqLabs focuses on user-centric design approaches.",
    "WerqLabs combines creativity with technology for better products.",
    
    "WerqLabs is located in Navi Mumbai, India.",
    "WerqLabs has offices in Mumbai, Bengaluru, and Dubai.",
    "WerqLabs works with clients across global markets.",
    "WerqLabs collaborates closely with clients to deliver solutions.",
    "WerqLabs focuses on innovation and continuous improvement.",
    "WerqLabs builds long-term relationships with clients.",
    "WerqLabs aims to bridge the gap between business and technology.",
    "WerqLabs delivers projects with fast turnaround times."
]

embeddings = model.encode(texts)  # Returns a numpy array
# print(embeddings)
# 
# [[-0.06681942  0.02552918 -0.05245946 ... -0.02625189  0.00493789
#    0.06243288]
#  [-0.01190417 -0.06289672  0.07373436 ... -0.05308472 -0.04357085
#    0.0353574 ]
#  [-0.02173978  0.07405403  0.0169607  ... -0.01395182 -0.02160488
#    0.01624599]]


#: To Do: 
#~Done....
# 5. Create a Vector Search Index on Atlas
# Go to Atlas UI → Your Cluster → Search → Create Search Index → JSON Editor, and paste:

# {
#   "fields": [
#     {
#       "type": "vector",
#       "path": "embedding",
#       "numDimensions": 384,
#       "similarity": "cosine"
#     }
#   ]
# }

#Storing the data into documents now.
# documents = [
#     {
#         "text": text,
#         "embedding": embedding.tolist(),  # Convert numpy array → plain list #MUST!!!!
#         "metadata": {"source": "data", "index": i}
#     }

#     for i, (text, embedding) in enumerate(zip(texts, embeddings))
# ]

# result = collection.insert_many(documents)
# print(f"Inserted {len(result.inserted_ids)} documents")



#Search
# def semantic_search(query_text: str, top_k: int = 3):
#     query_embedding = model.encode(query_text).tolist()

#     pipeline = [
#         {
#             "$vectorSearch": {
#                 "index": "vector_index",
#                 "path": "embedding",
#                 "queryVector": query_embedding,
#                 "numCandidates": 50,   # Candidates to consider (higher = more accurate)
#                 "limit": top_k         # Final results to return
#             }
#         },
#         {
#             "$project": {
#                 "text": 1,
#                 "metadata": 1,
#                 "score": { "$meta": "vectorSearchScore" },  # Similarity score
#                 "embedding": 0  # Exclude large vector from results
#             }
#         }
#     ]

#     return list(collection.aggregate(pipeline))

# #Actual Search
# results = semantic_search("Where is WerqLabs Located?")
# for r in results:
#     print(f"[{r['score']:.4f}] {r['text']}")


def search_query(query: str ):
    query_embedding = model.encode([query])[0] #this only accepts list
    results = collection.aggregate([
    {
        "$vectorSearch": {
        "queryVector": query_embedding.tolist(),
        "path": "embedding",
        "numCandidates": 100,
        "limit": 3,
        "index": "vector_index"
        }
    },
    {
        "$project": {
        "text": 1,
        "metadata": 1,
        "score": { "$meta": "vectorSearchScore" },
        "_id": 0
        }
    }
    ])
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print(f'For Query: {query}')
    for doc in results:
        print(f"{doc['score']:.4f} → {doc['text']}")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    return True



#   {
#     "$vectorSearch": {
#       "queryVector": query_embedding.tolist(),
#       "path": "embedding",
#       "numCandidates": 100,
#       "limit": 3,
#       "index": "vector_index"
#     }
#   }
# ])

# print("Raw Results::")
# print(results)
# print("-------------------------------------------------------------")

# print('Doc Results::')
# for doc in results:
#     print(doc)
#     print('---')
# print("-------------------------------------------------------------")



# Doc Results::
# {'_id': ObjectId('69c4ca80ae2ac11906338901'), 'text': 'WerqLabs is located in Navi Mumbai, India.', 'embedding': [rmeovd this], 'metadata': {'source': 'data', 'index': 56}}

# {'_id': ObjectId('69c4c71f5f206c9d3aa60f8a'), 'text': 'WerqLabs is located in Vashi.', 'embedding': [removed this], 'metadata': {'source': 'example', 'index': 0}}




search_query('Where is WerqLabs Located?')
search_query('What roles werqlabs offer?')
search_query('What cloud service does werqlabs use?')
search_query('Werqlabs works in which domain?')
search_query('werqlabs internship')



# For Query: Where is WerqLabs Located?
# 0.9335 → WerqLabs is located in Navi Mumbai, India.
# 0.9099 → WerqLabs is located in Vashi.
# 0.8811 → WerqLabs uses cloud platforms like AWS and Azure.

# For Query: What roles werqlabs offer?
# 0.8981 → WerqLabs offers roles in QA testing and software development.
# 0.8685 → WerqLabs helps companies scale teams quickly with skilled developers.
# 0.8681 → WerqLabs has expertise in multiple technology stacks and frameworks.

# For Query: What cloud service does werqlabs use?
# 0.9478 → WerqLabs uses cloud platforms like AWS and Azure.
# 0.9217 → WerqLabs offers services in software development and cloud computing.
# 0.8839 → WerqLabs ensures high-quality IT infrastructure for clients.

# For Query: Werqlabs works in which domain?
# 0.8606 → WerqLabs uses cloud platforms like AWS and Azure.
# 0.8590 → WerqLabs is located in Navi Mumbai, India.
# 0.8513 → WerqLabs works across multiple industries including healthcare and finance.


# For Query: werqlabs internship
# 0.9205 → WerqLabs offers internship opportunities in software development.
# 0.9107 → WerqLabs internships require knowledge of programming basics.
# 0.9095 → WerqLabs internships focus on real-world project experience.