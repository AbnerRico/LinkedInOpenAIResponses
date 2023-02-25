import openai

openai.organization = "[YOUR ORG ID]"
openai.api_key = "[YOUR OPENAI API KEY]"

MAX_TOKENS = 100
MODEL = "text-davinci-003"

def send_prompt(p):
  try:
    return openai.Completion.create(
            model=MODEL,
            prompt=p,
            max_tokens=MAX_TOKENS,
            temperature=0
          )
  except:
    return None