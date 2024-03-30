# General

- Use `is None` instead of `== None`
    - `== None` calls the __eq__ method of the object. `==` can return True or False depending on how __eq__ is
      implemented for custom objects.
    - `is None` checks if the value is exactly the None object. This is preferred.
- `while head:` or `if not head:` can be used instead of `while head is not None` or `if head is None:`
- `left, right = right, left` # swap without using temporary variable

# Int

- int("123")
- min() or max()
- abs()
- 2**3 # shortcut for math.pow(2, 3)

# String

- my_list = s.split(sep='')
- ''.join(my_list)
- s.find(key) or s.find(key, start, end)
- s[0:2]
- s += "def"
- str(123)
- s.replace(old, new)
- s.strip(' ')
- len(string)
- sorted(string) # can't use .sort() on immutable types # sorted returns list & not string.
- ''.join(sorted(string)) # convert the list which sorted gives to string. sorted returns list.
- s.upper() or s.lower()
- s.isalpha() or s.isdigit() or s.isalnum()
- "abc" * 3 # returns 'abcabcabc'
- ord(character) -> returns ascii value

# List

- [1] * n # returns [1, 1, 1, 1, 1]
- [1, 2, 3] * 2 # returns [1, 2, 3, 1, 2, 3]
- alphabets = [0] * 26 or alphabets = [0 for i in range(26)]
- append(x)
- list1.extend(list2) or list1 = [*list1, *list2]
- insert(index, element)
- list1.copy() # create copy of list
- remove(x) # raise errors if not exist
- pop() or pop(index)
- sort() or sort(reverse=True)
- arr.sort(key=lambda x: len(x)) # custom sort
- arr = [i for i in range(5)] # 1d array
- arr = [[0] * 4 for i in range(4)] # 2d array
- arr.reverse()
- reversed(array)
- len(list)
- arr[-1] # last element
- arr[1:3] # exclude 3rd index
- for i in range(len(nums)): pass # use index
- for n in nums: # use value
- for i, n in enumerate(nums): # use index & value
- for n1, n2 in zip(nums1, nums2): # loop through multiple arrays simultaneously with unpacking
- sorted(my_list) # returns new sorted array
- min() or max()
- sum()
- dict = {key1: [], key2: []} # list(dict.values()) to convert the values to a list. values will be list of list.

# Set

- set() or set(my_list)# create empty set # {} creates a dictionary
- add(elem)
- set = {1,2,3}
- my_set = { i for i in range(5) }
- len(my_set)
- 1 in my_set
- my_set.update(set or list or tuple or string)
- remove(element) # raise key error if not present
- discard(element) # does not raise error
- | -> union()
- & -> intersection()
-
    - -> difference()
- ^ -> symmetric_difference()

# Dictionary

- my_map = {}
- my_map['key'] # raise KeyError if key doesn't exist
- my_map = { i: 2*i for i in range(3) }
- get(key) or get(key, 'default') # does not raise error if key not exist
- len(my_map)
- pop(key) or pop(key, 'default') # pop without default raise key error
- keys()
- values()
- items() # key and value
- dict1.update(dict_2)

# Empty

- len(stack)==0
- not stack # return true if stack is empty
- works for most datastructures

# Slicing

```python
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(my_list[2:6])  # Output: [2, 3, 4, 5]
print(my_list[:5])  # Output: [0, 1, 2, 3, 4]
print(my_list[5:])  # Output: [5, 6, 7, 8, 9]
print(my_list[0:10:2])  # Output: [0, 2, 4, 6, 8]
print(my_list[::-1])  # Output: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

my_string = "Hello, World!"
print(my_string[7:12])  # Output: "World"
print(my_string[::2])  # Output: "Hlo ol!"
```

# Tuple

- my_tuple[0] or my_tuple[-1]
- my_map = { (1,2): 3 } # can be used as keys for map
- my_set.add((1, 2)) # can be used as keys for set
- my_tuple.count(value)
- my_tuple.index(value) or my_tuple.index(value, start, end) # gives key error if not found
- value in my_tuple

# Stack

- stack = []
- list.append(element) # Push (add an element to the top of the stack)
- element = list.pop() # Pop (remove the top element from the stack)
- element = list[-1] # Peek (get the top element without removing)

# Deque

- my_deque = deque()
- append(x): Add x to the right side.
- appendleft(x): Add x to the left side.
- pop(): Remove and return an element from the right side.
- popleft(): Remove and return an element from the left side.
- extend(iterable): Extend the right side with elements from the iterable.
- extendleft(iterable): Extend the left side with elements from the iterable, added in reverse order.
- reverse(): Reverse the elements of the deque in-place.
- rotate(n): Rotate the deque n steps to the right. If n is negative, rotate to the left.

# Counter

- char_count = Counter("banana") or Counter(my_list)
- print(char_count)  # Output: Counter({'a': 3, 'b': 1, 'n': 2})
- elements(): Return an iterator over elements repeating each as many times as its count.
- most_common([n]): Return a list of the n most common elements and their counts.
- [element for element, count in counter.most_common(k)] # most_common returns tuple. use list comprehension to get only
  keys.
- subtract([iterable-or-mapping]): Subtract elements count from the counter.

# Default Dict

- Usage: defaultdict(default_factory_function), where default_factory_function provides the default value for a new key.
- dict = defaultdict(int) # int() is inbuilt a function which gives 0
- dict = defaultdict(str) # str() is inbuilt function with empty value
- dict = defaultdict(list) # list() is inbuilt function with empty list
- dict = defaultdict(set) # set() is inbuilt function with empty set
- dict = defaultdict(lambda: "specific default value") # lambda is a function. if we give just a string without lambda,
  it raises error.

# Heapq

- min_heap = [] # by default, it's minheap. uses array under the hood.
- heapq.heapify(x): Transform list x into a heap.
- while len(minHeap): print(heapq.heappop(minHeap)) # tranverse min heap
- heapq.heappush(heap, item): Push a new item onto the heap.
- heapq.heappop(heap): Pop the smallest item off the heap.
- min_heap[0] # minimum value always at index 0
- heapq.heappushpop(heap, item): Push item on the heap and then pop and return the smallest item from the heap.
- heapq.nsmallest(n, iterable): Return a list with the n smallest elements from the dataset. iterable can be a list.
- heapq.nlargest(n, iterable): Return a list with the n largest elements from the dataset.
- max_heap logic -> use negative values for all operations.
    - maxHeap = []
    - heapq.heappush(maxHeap, -3)
    - heapq.heappush(maxHeap, -2)
    - heapq.heappush(maxHeap, -4)
    - print(-1 * maxHeap[0])
    - while len(maxHeap): print(-1 * heapq.heappop(maxHeap))

# Priority Queue

- implemented by heapq

```python
import heapq

priority_queue = []
heapq.heappush(priority_queue, (1, 'Task 1'))
heapq.heappush(priority_queue, (3, 'Task 3'))
heapq.heappush(priority_queue, (2, 'Task 2'))
highest_priority = heapq.heappop(priority_queue)
print(f"Highest priority task: {highest_priority}")

# The rest of the queue
print("Remaining tasks in the queue:")
for task in priority_queue:
    print(task)
```

- tuples are compared using lexicographical ordering, meaning they are compared item by item starting from the first.
- the comparison starts with the first element of each tuple. If the first elements are equal, Python compares the
  second elements, and so on.

# Comprehensions

- arr = [x + 1 for x in range(5)]
- max_heap = [(12, [0, 1])]
    - [point for (dist, point) in max_heap]
- evens = [number for number in range(50) if number % 2 == 0]

```python
options = ["any", "w", "zh"]
string_start_with_a_and_end_with_y = [
    string
    for string in options
    if len(string) >= 2
    if string[0] == "a"
    if string[-1] == "y"
]
```

```python

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened1 = [num for row in matrix for num in row]
# exterior for loop is first. then, interior for loops.

flattened2 = []
for row in matrix:
    for num in row:
        flattened2.append(num)

```

```python
categories = [
    "Even" if x % 2 == 0 else "Odd" for x in range(10)
]
```

```python
def square(x):
    return x ** 2


squared_numbers = [square(x) for x in range(10)]
```

```python
# dictionary comprehension
pairs = [("a", 1), ("b", 2), ("c", 3)]
my_dict = {k: v for k, v in pairs}
```

```python
# set comprehension
nums = [1, 1, 2]
unique_squares = {x ** 2 for x in nums}
```

```python
# generator comprehension
sum_of_squares = sum(x ** 2 for x in range(1000000))
```

# Custom sorting

```python
people = [('Alice', 30), ('Bob', 25), ('Charlie', 35)]
sorted_people1 = sorted(people, key=lambda person: person[1])  # Output will be sorted by age
people.sort(key=lambda person: person[1])  # Output will be sorted by age

people = [('Alice', 30), ('Bob', 25), ('Charlie', 30)]
sorted_people2 = sorted(people, key=lambda person: (person[1], person[0]))  # Output will be sorted by age, then by name
people.sort(key=lambda person: (person[1], person[0]))  # Output will be sorted by age, then by name

numbers = [3, 1, 4, 1, 5, 9, 2]
sorted_numbers = sorted(numbers, reverse=True)
numbers.sort(reverse=True)
```

# Itertools

- itertools.permutations(items, 2): Generate all possible permutations of length 2.
- itertools.combinations(items, 2): Generate all possible combinations of length 2.

# Math

- math.sqrt(x): Return the square root of x.
- math.pow(x, y): Return x raised to the power of y.
- math.floor(3 / 2)
- math.ceil(3 / 2)
- math.sqrt(2)

# Float

- float("inf")
- float("-inf")

# Error Handling

```python
x, y = 0, 0
try:
    result = x / y
except ZeroDivisionError:
    print("Error: You can't divide by zero!")
else:
    print("Result:", result)
finally:
    print("Executing finally block...")
```

# File Operations

```python
with open('output.txt', 'w') as file:
    file.write('Hello, world!')

with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
```

# Functional Programming

```python
# filter() example: Filter even numbers
nums = [1, 2, 3, 4, 5]
even_nums = list(filter(lambda x: x % 2 == 0, nums))
print(even_nums)  # Output: [2, 4]

# map() example: Square each number
squared_nums = list(map(lambda x: x ** 2, nums))
print(squared_nums)  # Output: [1, 4, 9, 16, 25]

# reduce() example: Sum all numbers
from functools import reduce

total = reduce(lambda x, y: x + y, nums)
print(total)  # Output: 15
```

# Functions within Functions

- doesn't need self syntax
- can use variables from outer function. no need to pass variables

```python
def generate_parenthesis(n: int) -> list[str]:
    def generate(current, open_count, close_count):
        if len(current) == n * 2:
            result.append(current)
            return

        if open_count < n:
            generate(current + "(", open_count + 1, close_count)
        if close_count < open_count:
            generate(current + ')', open_count, close_count + 1)

    result = []
    generate("", 0, 0)
    return result
```

# Regex

```python
import re

text = "The rain in Spain"
# Find all matches of 'ai' in the text
matches = re.findall('ai', text)
print(matches)  # Output: ['ai', 'ai']

# Split the text at each space
split_text = re.split('\s', text)
print(split_text)  # Output: ['The', 'rain', 'in', 'Spain']

# Replace all spaces with a hyphen
replaced_text = re.sub('\s', '-', text)
print(replaced_text)  # Output: The-rain-in-Spain
```

# Lambda

```python
# A simple lambda function to add two numbers
add = lambda x, y: x + y
print(add(5, 3))  # Output: 8

# Lambda function within a list comprehension
squared = [(lambda x: x ** 2)(x) for x in range(10)]
print(squared)  # Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

# Generator

- Generator expressions: (x**2 for x in range(10))

# Functions

```python
def my_fun(n, m):
    return n * m


print(my_fun(3, 4))


# Nested functions have access to outer variables
def outer(a, b):
    c = "c"

    def inner():
        return a + b + c

    return inner()


print(outer("a", "b"))


# Can modify objects but not reassign
# unless using nonlocal keyword
def double(arr, val):
    def helper():
        # Modifying array works
        for i, n in enumerate(arr):
            arr[i] *= 2

        # will only modify val in the helper scope
        # val *= 2

        # this will modify val outside helper scope
        nonlocal val
        val *= 2

    helper()
    print(arr, val)


nums = [1, 2]
val = 3
double(nums, val)
```

# Class

```python
class MyClass:
    # Constructor
    def __init__(self, nums):
        # Create member variables
        self.nums = nums
        self.size = len(nums)

    # self key word required as param
    def get_length(self):
        return self.size

    def get_double_length(self):
        return 2 * self.get_length()


myObj = MyClass([1, 2, 3])
print(myObj.get_length())
print(myObj.get_double_length())
```

# Syntax

```python
n, m, z = 0.125, "abc", False  # multi assignment
n += 1

if n > 2:
    n -= 1
elif n == 2:
    n *= 2
else:
    n += 2

for i in range(5, 1, -1):
    pass

print(5 / 2)  # decimal
print(5 // 2)  # int result

print(-3 // 2)  # -2 -> most languages round towards 0 for negative. but python moves away from 0 & gives smaller value
print(int(-3 / 2))  # -1 -> fixes the above issue and moves towards 0 for negatives.

import math

print(-10 % 3)  # 2. most languages round towards 0 for negative. but python moves away from 0
print(math.fmod(-10, 3))  # -1.0 -> fixes the above issue and moves towards 0 for negatives.
```