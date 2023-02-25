import csv
import pandas as pd
from bs4 import BeautifulSoup
from InboxMessage import InboxMessage
from openAIApi import send_prompt

# This script generates responses to all my LinkedIn inbox messages using AI provided by OpenAI

# Load the CSV file with all the messages data in my linkedIn inbox.
df = pd.read_csv("assets/messages.csv")

# Remove "Abner" and other words from Message content.
df['CONTENT'] = df['CONTENT'].str.replace('[your name here]', '', regex=True)
df['CONTENT'] = df['CONTENT'].str.replace('%FIRSTNAME%', '', regex=True)

# Strip all the html code and other noise from every message. this could cause an unpredictable and unwanted response from OpenAI
messages = []
for index, row in df.iterrows():
    try:
        row['CONTENT'] = BeautifulSoup(row['CONTENT'], 'html.parser').get_text(strip=False)
        m = InboxMessage(content=row['CONTENT'], sender=row['FROM'])

        # Analize the content of the message and get polarity scores, 
        # We need to make sure OpenAI gets to answer only potentially positive messages ;)
        m.analize()
        messages.append(m)
    except:
        continue

# Get only the messages with a certain positiveness
# We don´t want OpenAI to get angry with us.
POSITIVENESS_THRESHOLD = .3
positiveMessages = [m for m in messages if m.sentimentAnaliysis['pos'] >= POSITIVENESS_THRESHOLD]

print(f"{len(positiveMessages)} mensajes con puntuación positiva >= {POSITIVENESS_THRESHOLD}")

# Profile summary to feed the openAi prompt and get the most relevant context on every response.
PROFILE_SUMMARY = ("full stack web developer")

# Send every potentially positive message to OpenAI asking it to respond as a job position interviewee.
for pm in positiveMessages:
    openAIResponse = send_prompt(f"Act as a job position interviewee {PROFILE_SUMMARY} and respond the following: {pm.content}")
    if openAIResponse:
        pm.SetResponse(openAIResponse["choices"][0]["text"].replace('\n', ' ').replace('\r', ''))
    else:
        # If we can't get an answer from openAI just generate a generic offer rejection message.
        pm.SetResponse(("Thank you very much for offering me the position. "
                        "I sincerely appreciate the offer and your interest in hiring me. "
                        "After much consideration, I have decided to accept another role that will offer me more "
                        "opportunities to pursue my interests"))

# Write all the responses with te corresponding messages and senders.
# We could just consume the linkedIn API and automatically respond every message with the generated response from OpenAI
with open('assets/output.csv', 'w', newline='') as output:
     wr = csv.writer(output, quoting=csv.QUOTE_ALL)
     wr.writerow(["HUMAN","HUMAN_MESSAGE","AI", "AI_RESPONSE"])  
     for m in positiveMessages:
        wr.writerow([m.sender, m.content, "OpenAI text-davinci-003", m.response])