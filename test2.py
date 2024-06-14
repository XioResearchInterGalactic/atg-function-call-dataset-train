# expression = "from random import randint\n\n# Define the probability of getting 6 on a single toss\np_six = 1/6\n\n# Define the probability of getting 6 twice in a row\np_double_six = p_six * p_six\n\nresult = p_double_six"
expression = "x=5\ny=6\nresult = x+y\nprint(result)"
result = {}
result = exec(expression, result)
# %%
print(result)