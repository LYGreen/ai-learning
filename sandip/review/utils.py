
# function to greet
def greet(name):
	return f"Hello, {name}!"

# function to calculate average of numbers
def calculate_average(numbers):
	return sum(numbers) / len(numbers)

# function to add all the number and give total
def add(total):
	return sum(total)

""" why __name__ == "__main__":?

 //gaurd clause - control structure that ensure the code only runs when the file is executed directly.

If we don't have a guard, every time you import the file, the code inside the __name__ == "__main__": run right away even though you only wanted to call functions explicitely. in short it's a side effect of importing. with the guard, you stop the automatic execution, so your function just stay ready to use, but don't run unless you explicitly trigger them. """


""" why even care to add guard? 

for clarity and predictability. with the guard, you defile exactly when code runs. without it, you risk unintentional side effects like functions that relies on file paths or user input triggering immediately. The guard ensures your code is modular, it's only active when you intend it, which reduces bugs and keeps your logic scoped so it's all about safer, more intentional structure. """

""" when to add guard and when not to? 

add guard whenever you want the file to be reusable so if your defining functions, classes, or utilities that might be imported elsewhere, always put them inside. Don't add a guard if the file is purely a script like a one-off program you run to produce output because in that case, you do want the code to run as soon as you execute the file. """

""" what if i forget to add guard?

- runs imediately when someone imports the file
- unexpected behaviour for eg: trying to read files, connect to a database or print clutter
- code becomes brittle and harder to reuse.
"""

""" where is the error going to be seen if forgot to add guard?

- terminal or console
- it tries to log something or perform side effects like accessing a file or network, or not on terminal or console. 
"""

if __name__ == "__main__": 
	print(greet("sandip")) 
	print(calculate_average([4, 7, 10, 7])) 
	print(add([7, 6, 9, 10]))
