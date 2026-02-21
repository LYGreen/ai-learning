class Animal:
    name = "Lion" # class attribute

    def __init__(self, age):
        self.age = age # instance attribute
        
    def show(self): # instance method
        print(f"This is a {self.name} and age is {self.age}")
    
    @classmethod
    def show_class_attribute(cls): # class method
        print(f"This is a class attribute: {cls.age}")

    @staticmethod
    def show_static_method(): # static method
        print("This is a static method")

obj = Animal(10)
obj.show()