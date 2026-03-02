# Bank Account Management System

import json
import random
import string
from pathlib import Path

# Create a class for the bank account
class BankAccount:
    # Define the database and data
    database = "data.json"
    data = []

    # Check if the file exists and load existing data
    try:
        if Path(database).exists():
            with open(database) as file:
                data = json.load(file)  # load full list from JSON file
        else:
            print("File does not exist") # start with empty list if file does not exist
    except Exception as err:
        print(f"Error: {err}")
    
    # Update the account
    @classmethod
    def __updateAccount(cls):
        with open(cls.database, "w") as file:
            file.write(json.dumps(BankAccount.data))
    
    # Generate a random account number
    @classmethod
    def __accountGenerate(cls):
        # Generate a random account number
        alpha = random.choices(string.ascii_letters, k=3)
        # Generate a random number
        num = random.choices(string.digits, k=3)
        # Generate a random special character
        spchar = random.choices("!@#$%^&*()", k=1)
        # Combine the random account number
        id = alpha + num + spchar
        # Shuffle the random account number
        random.shuffle(id)
        # Convert the list to a string
        return "".join(id)

    # Create a bank account
    def createAccount(self):
        # Take the user's input
        info = {
            "name": input("Enter your name: "),
            "age": int(input("Enter your age: ")),
            "email": input("Enter your email: "),
            "pin" : int(input("Enter your 4 digit pin: ")),
            "account_number": BankAccount.__accountGenerate(),
            "balance": 0
        }
        # Check if the user is eligible to create a bank account
        if info['age'] < 18 or len(str(info['pin'])) != 4:
            print("You are not eligible to create a bank account")
        else:
            print("Account created successfully")
            for i in info:
                print(f"{i}: {info[i]}")
            print("Please note down your account number")

        # Update the account
        BankAccount.data.append(info)
        BankAccount.__updateAccount()
    
    # Deposit money
    def depositMoney(self):
        account_number = input("Enter your account number: ")
        pin = int(input("Enter your pin: "))

        print(BankAccount.data)

user = BankAccount()
# Print the menu options
print("Press 1 for creating a bank account")
print("Press 2 for depositing money")
print("Press 3 for withdrawing money")
print("Press 4 for checking details")
print("Press 5 for updating details")
print("Press 6 for deleting the account")

# Take the user's response
check = int(input("Tell your response: "))

if check == 1:
    # Create a bank account
    user.createAccount()
if check == 2:
    # Deposit money
    user.depositMoney()