# student dict
# dict or dictonary is basically a collection of key-value pair
student = {
    "name": "Sandip",
    "course": ["ML", "Python", "Data Science"], # course list inside a dict
    "Level": 2
}

#adding items: key and values in the dict student.
student["completed"] = ["dict"] # adding a completed list in the student dict

for a, b in student.items():
    print(a, b)

# dictionary method: .items() .key() .values() .get() .pop() .update() most common dictonary method.