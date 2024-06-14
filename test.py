#%%
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# Two lists of sentences
sentences1 = [
    "[FUNCTION_CALL] {\n  \"name\": \"get_web_search_result\",\n  \"arguments\": {\n    \"query\": \"weather in Toronto\"\n  }\n}</s>",
    "[FUNCTION_CALL] {\n  \"name\": \"get_web_search_result\",\n  \"arguments\": {\n    \"query\": \"weather in Toronto\"\n  }\n}</s>",
    "[FUNCTION_CALL] {\n  \"name\": \"get_web_search_result\",\n  \"arguments\": {\n    \"query\": \"weather in Toronto\"\n  }\n}</s>",
]

sentences2 = [
    "[FUNCTION_CALL] {\n  \"name\": \"get_web_search_result\",\n  \"arguments\": {\n    \"query\": \"weather in Toronto\"\n  }\n}</s>",
    "[FUNCTION_CALL] {\n  \"name\": \"get_python_math_result\",\n  \"arguments\": {\n    \"expression\": \"#User wants to know the difference between 15 and 20\\n\\n#Importing numpy library\\nimport numpy as np\\n\\n#Calculating the difference\\ndifference = np.abs(15 - 20)\\n\\n#Printing the result\\nprint(difference)\"\n  }\n}</s>",
    "[FUNCTION_CALL] {\n  \"name\": \"get_web_search_result\",\n  \"arguments\": {\n    \"query\": \"current weather in canada\"\n  }\n}</s>",
]

# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(
        sentences1[i], sentences2[i], cosine_scores[i][i]
    ))
# %%
