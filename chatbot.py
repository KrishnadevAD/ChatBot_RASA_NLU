from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
import warnings
warnings.filterwarnings("ignore")

# Define training data
training_data = load_data("data.json")

# Define model configuration
config = RasaNLUModelConfig({"pipeline": "tensorflow_embedding","language":"en_core_web_sm"})

# Create a trainer
trainer = Trainer(config)

# Train the model
interpreter = trainer.train(training_data)

# Define a dictionary to store account information
accounts = {"12345": {"name": "krishna dev", "balance": 1000},
            "67890": {"name": "simanta karki", "balance": 500}}

# Define a function to handle the user's message
def handle_message(message):
    # Use the interpreter to parse the user's message
    parsed_message = interpreter.parse(message)
    # Extract the intent and entities from the parsed message
    intent = parsed_message["intent"]["name"]
    entities = parsed_message["entities"]
    # Initialize an empty response
    response = ""

    # Handle different intents
    if intent == "greet":
        response = "Hello! How can I help you today?"
    elif intent == "goodbye":
        response = "Goodbye! Have a great day."
    elif intent == "check_balance":
        account_number = None
        name = None
        for entity in entities:
            if entity["entity"] == "account_number":
                account_number = entity["value"]
            if entity["entity"] == "name":
                name = entity["value"]
        if account_number in accounts:
            account_info = accounts[account_number]
            if name is not None and name != account_info["name"]:
                response = "Sorry, the account number and name you provided do not match."
            else:
                response = f"Dear {account_info['name']}, your account number {account_number} has a balance of Rs. {account_info['balance']}."
        else:
            response = "Sorry, I couldn't find an account with that number."
    else:
        response = "I'm sorry, I didn't understand what you meant. Could you please rephrase your question?"
    return response



query=""
while(query!="end"):
    query=input("Enter command:\n")
    # Test the function with a message
    print(handle_message(query))


