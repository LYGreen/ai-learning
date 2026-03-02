# Dunder methods

class Employee:
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary

    def __str__(self):
        return f"Hi {self.name}, How are you?\nYour salary is {self.salary}"
               

obj = Employee("John", 30, 10000)
print(obj)