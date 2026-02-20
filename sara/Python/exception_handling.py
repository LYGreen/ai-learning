# Exception Handling

a = int(input("Enter a number: "))

try:
    print(10/a)
except Exception as err:
    print(f"There is an error: {err}")
else:
    print("No exception occurred")
finally:
    print("I will run this no matter what")

print("Ok I have done the division")

# Custom Exception Handling

age = int(input("Enter your age: "))

try:
    if age < 10 or age > 18:
        raise ValueError("Your age is not in the range of 10 to 18")
    else:
        print("Welcome to the club")
except ValueError as err:
    print(f"There is an error: {err}")

print("Club will open after 10 minutes")

# git commit -m "Added exception handling"