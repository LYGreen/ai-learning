# Syntax of Inheritance

class parent:
    def speak(self):
        print("I can speak!")

class child(parent):
    pass

# Constructor in Inheritance

class parent:
    def __init__(self, name):
        self.name = name

class child(parent):
    def display(self):
        print(f"My name is {self.name}")