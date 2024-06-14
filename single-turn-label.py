#%%
from datasets import load_dataset
from dotenv import load_dotenv
import os
from labelstudio import LabelStudio
from tqdm import tqdm

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
LABEL_STUDIO_PROJECT_ID = os.getenv("LABEL_STUDIO_WEB_SEARCH_SINGLE_TURN_PROJECT_ID")
dataset = load_dataset("MerlynMind/RAG_Current_Events_v1_20240220", token=HF_TOKEN, split="train")
ls = LabelStudio(
    url=LABEL_STUDIO_URL,
    api_key=LABEL_STUDIO_API_KEY,
    project_id=LABEL_STUDIO_PROJECT_ID
)

#%%
i = 1
for row in tqdm(dataset):
    ls.log_single_turn_web_search(
        id=i,
        row=row
    )
    i += 1

# %%
tool_name = "get_web_search_result"
argument_label = "query"
web_search_items = [
    {'user_message': "Compare sharks and tacos", "search_terms": ["sharks", "tacos"]},
    {'user_message': "what are the top news today", "search_terms": ["top news today"]},
    {'user_message': "what are the top restaurants in New York", "search_terms": ["top restaurants in New York"]},
    {'user_message': "Suggest some TV shows to watch", "search_terms": ["TV shows to watch"]},
    {'user_message': "When is the next Super Bowl", "search_terms": ["when is the next Super Bowl"]},
    {'user_message': "How to screenshot in Mac", "search_terms": ["how to screenshot in mac"]},
    {'user_message': "When is Father's Day", "search_terms": ["father's day date"]},
    {'user_message': "How many ounces in a pound", "search_terms": ["how many ounces in a pound"]},
    {'user_message': "How many teaspoons in a tablespoon", "search_terms": ["how many teaspoons in a tablespoon"]},
    {'user_message': "How long to boil eggs", "search_terms": ["how long to boil eggs"]},
    {'user_message': "How to make pancakes", "search_terms": ["how to make pancakes"]},
    {'user_message': "When does the DST time change", "search_terms": ["when does the DST time change"]},
    {'user_message': "When is the next taylor swift concert", "search_terms": ["next taylor swift concert date"]},
    {'user_message': "How to solve a Rubik's Cube", "search_terms": ["how to solve a Rubik's Cube"]},
    {'user_message': "What does smh mean", "search_terms": ["smh meaning"]},
    {'user_message': "When does summer start", "search_terms": ["summer start date"]},
    {'user_message': "what are synonyms", "search_terms": ["what are synonyms"]},
    {'user_message': "What do you understand by the term \"foundation of Arts\"", "search_terms": ["foundation of arts meaning"]},
    {'user_message': "List the moons of Jupiter", "search_terms": ["list moons of Jupiter"]},
    {'user_message': "Can you define the term accelaration in simple words", "search_terms": ["accelaration meaning"]},
    {'user_message': "who is known as the father of modern physics", "search_terms": ["father of modern physics"]},
    {'user_message': "can you name the first human to walk on the moon", "search_terms": ["first human to walk on the moon"]},
    {'user_message': "book a flight", "search_terms": ["book a flight"]},
    {'user_message': "who invented computer", "search_terms": ["computer inventor"]},
    {'user_message': "what was the ancient city of Rome known for", "search_terms": ["what was the ancient city of Rome known for"]},
    {'user_message': "whats the address of Merlyn Mind", "search_terms": ["merlyn mind address"]},
    {'user_message': "how to open an account in the bank", "search_terms": ["how to open an account in the bank"]},
    {'user_message': "who is donald trump", "search_terms": ["donald trump"]},
    {'user_message': "who is the current president of United States", "search_terms": ["current president of United States"]},
    {'user_message': "Which planet is the closest in size to Earth?", "search_terms": ["which planet is the closest in size to Earth"]},
    {'user_message': "Write a template email telling parents to remember to return their signed permission slips.", "search_terms": []},
    {'user_message': "Make a bulleted list including a list of animals starting with \"A\"", "search_terms": ["animals starting with A"]},
    {'user_message': "What is the tallest building in the world", "search_terms": ["tallest building in the world"]},
    {'user_message': "Which type of animal lays eggs: mammals or reptiles?", "search_terms": ["do mammals lay eggs", "do reptiles lay eggs"]},
    {'user_message': "What is the best pizza restaurant in Toronto and how is the weather there", "search_terms": ["best pizza restaurant in Toronto", "weather in Toronto"]},
    {'user_message': "Who was the last wife of the current president of United States", "search_terms": ["current president of United States", "last wife of Joe Biden"]},
    {'user_message': "When was the lead singer of Green Day born?", "search_terms": ["Green Day lead singer", "when was Billie Joe Armstrong born"]},
    {'user_message': "What is the top-rated hiking trail in Yosemite National Park, and what is the current entry fee?", "search_terms": ["top rated hiking trail in Yosemite National Park", "current entry fee in Vernal and Nevada Falls"]},
    {'user_message': "who are you", "search_terms": []},
    {'user_message': "what can you do", "search_terms": []},
    {'user_message': "write a story", "search_terms": []},
    {'user_message': "write a poem", "search_terms": []},
    {'user_message': "explain modern physics in simple terms", "search_terms": ["modern physics"]},
    {'user_message': "What are the key principles of smart investing?", "search_terms": ["key principles of smart investing"]}
]
for row in tqdm(web_search_items):
    ls.log_single_turn(
        row['user_message'],
        tool_name,
        argument_label,
        row['search_terms']
    )
# %%
# weather
    # 
# translate
    # 
# math
    # 
# time
    # What time is it in Australia
    # What time is it in New York
    # What is the time in EST
# Stock exchange rates
# Currency rates