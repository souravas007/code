# General

- Use `is None` instead of `== None`
  - `== None` calls the **eq** method of the object. `==` can return True or False depending on how **eq** is
    implemented for custom objects.
  - `is None` checks if the value is exactly the None object. This is preferred.
- `while head:` or `if not head:` can be used instead of `while head is not None` or `if head is None:`
- `left, right = right, left` # swap without using temporary variable

# Int

- int("123")
- min() or max()
- abs()
- 2\*\*3 # shortcut for math.pow(2, 3)

# String

- strings are immutable. any operation that modifies a string, will return a new string.
- my_list = s.split() or my_list = s.split(sep='#')
- ''.join(my_list)
- s.find(key) or s.find(key, start, end) # returns -1 if not found. else index
- s[0:2]
- s += "def" # doesn't modify inplace. Equivalent to s = s + "def"
- str(123)
- s.replace(old, new) # returns new string
- s.strip(' ') # returns new string
- len(string)
- sorted(string) # can't use .sort() on immutable types # sorted returns list & not string.
- ''.join(sorted(string)) # convert the list which sorted gives to string. Sorted returns list.
- s.upper() or s.lower()
- s.isalpha() or s.isdigit() or s.isalnum()
- "abc" \* 3 # returns 'abcabcabc'
- ord(character) -> returns ascii value
- arr = [0]\*26;
- arr[ord(str[i]) - ord('a')] += 1 # get count of all characters in string

# List

- [1] \* n # returns [1, 1, 1, 1, 1]
- [1, 2, 3] \* 2 # returns [1, 2, 3, 1, 2, 3]
- alphabets = [0] \* 26 or alphabets = [0 for i in range(26)]
- append(x)
- list1 = [*list1, *list2]
- insert(index, element)
- list1.copy() or - new_list = old_list[:] # create copy of list
- remove(x) # raise errors if not exist
- pop() or pop(index) # remove last element or index
- sort() or sort(reverse=True)
- arr.sort(key=lambda x: len(x)) # custom sort using length of element in ascending order. len() exists in strings, not in integers.
- arr = [i for i in range(5)] # 1d array
- arr = [[0] \* 4 for i in range(4)] # 2d array
- arr.reverse()
- reversed(array)
- len(list)
- arr[-1] # last element
- arr[1:3] # exclude 0th & 3rd index
- for i in range(len(nums)): pass # use index
- for n in nums: # use value
- for i, n in enumerate(nums): # use index & value
- for n1, n2 in zip(nums1, nums2): # loop through multiple arrays simultaneously with unpacking
- sorted(my_list) # returns new sorted array
- min() or max()
- sum()
- dict = {"key1": [], "key2": []} # list(dict.values()) to convert the values to a list. values will be list of lists: [[], []]

# Set

- set() or set(my_list)# create empty set. Note: {} creates a dictionary
- set = {1,2,3}
- add(elem)
- my_set = { i for i in range(5) }
- len(my_set)
- 1 in my_set
- my_set.update(set or list or tuple or string)
- remove(element) # raise key error if not present
- discard(element) # does not raise error
- union_set = set1 | set2 # union()
- intersection_set = set1 & set2 # intersection()
- difference_set = set1 - set2 # difference() - all elements in a, but not in b
- sym_diff_set = set1 ^ set2 # symmetric_difference() - all elements in a or b, but not in both

# Dictionary

- my_map = {}
- my_map = {'(': ')', '{': '}', '[': ']'}
- my_map['key'] # raise KeyError if key doesn't exist
- my_map = {i: 2\*i for i in range(3)}
- value = my_map.get('key') or my_map.get('key', 'default') # returns 'None' or 'default' if key does not exist
- len(my_map)
- popped_value = my_map.pop('key') or my_map.pop('key', 'default') # pop without default raises KeyError
- keys = my_map.keys()
- values = my_map.values()
- items = my_map.items() # returns key-value pairs
- merged_dict = {**x, **y} or merged_dict = x | y # merge two dictionaries

# Empty

- len(stack) == 0
- not stack # returns True if stack is empty. Works for most data structures

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

- my_tuple = (1, 2, 3)
- my_tuple[0] or my_tuple[-1]
- my_map = {(1, 2): 3} # tuples can be used as keys in a map
- my_set.add((1, 2)) # tuples can be used as keys in a set
- my_tuple.count(value)
- my_tuple.index(value) or my_tuple.index(value, start, end) # raises ValueError if not found
- value_in_tuple = value in my_tuple # returns boolean

# Stack

- stack = []
- stack.append(element) # Push (add an element to the top of the stack)
- element = stack.pop() # Pop (remove the top element from the stack)
- top_element = stack[-1] # Peek (get the top element without removing)
- stack[-1] < stack[-2] # Compare values in the stack

# Deque

- from collections import deque
- my_deque = deque()

- my_deque.append(x): Add x to the right side.
- my_deque.appendleft(x): Add x to the left side.
- element = my_deque.pop() # Remove and return an element from the right side
- element = my_deque.popleft() # Remove and return an element from the left side
- my_deque.extend(iterable) # Extend the right side with elements from the iterable
- my_deque.extendleft(iterable) # Extend the left side with elements from the iterable, added in reverse order
- my_deque.reverse() # Reverse the elements of the deque in-place
- my_deque.rotate(n) # Rotate the deque n steps to the right. If n is negative, rotate to the left

# Counter

- from collections import Counter
- char_count = Counter("banana") or Counter(my_list)
- print(char_count) # Output: Counter({'a': 3, 'b': 1, 'n': 2})

- elements = list(char_count.elements()) # Convert the iterator to a list. Output: ['b', 'a', 'a', 'a', 'n', 'n'].
- common_elements = char_count.most_common(n) # Return a list of the n most common elements and their counts
- top_keys = [element for element, count in char_count.most_common(k)] # most_common returns tuple. Use list comprehension to get only keys
- char_count.subtract([iterable-or-mapping]) # Subtract elements count from the counter

# Default Dict

- from collections import defaultdict
- Usage: defaultdict(default_factory_function), where default_factory_function provides the default value for a new key.
- dict = defaultdict(int) # int() is inbuilt a function that returns 0
- dict = defaultdict(str) # str() is inbuilt function that returns empty string
- dict = defaultdict(list) # list() is inbuilt function that returns an empty list
- dict = defaultdict(set) # set() is inbuilt function that returns an empty set
- dict = defaultdict(lambda: "specific default value") # lambda is a function. if we give just a string without lambda, it raises error.

# Heapq

- import heapq # not part of collections
- min_heap = [] # by default, it's a min-heap. Uses an array under the hood.
- heapq.heapify(x): Transform list x into a heap.
- while len(min_heap): print(heapq.heappop(min_heap)) # Traverse and pop elements from the min-heap
- heapq.heappush(min_heap, item): Push a new item onto the heap.
- smallest_item = heapq.heappop(min_heap) # Pop the smallest item off the heap
- min_value = min_heap[0] # The minimum value is always at index 0 in a min-heap
- heapq.heappushpop(min_heap, item) # Push item onto the heap and then pop and return the smallest item from the heap
- smallest_n = heapq.nsmallest(n, iterable) # Return a list with the n smallest elements from the dataset. iterable can be a list.
- largest_n = heapq.nlargest(n, iterable) # Return a list with the n largest elements from the dataset
- Max-heap logic: use negative values for all operations to simulate a max-heap
  - maxHeap = []
  - heapq.heappush(maxHeap, -3)
  - heapq.heappush(maxHeap, -2)
  - heapq.heappush(maxHeap, -4)
  - print(-1 \* max_heap[0]) # To get the actual max value, multiply by -1
  - while len(max_heap):
    print(-1 \* heapq.heappop(max_heap)) # Traverse and pop elements from the max-heap

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

- Note: Tuples are compared using lexicographical ordering, meaning they are compared item by item starting from the first.
- The comparison starts with the first element of each tuple. If the first elements are equal, Python compares the second elements, and so on.

# Comprehensions

## List comprehensions

- arr = [x + 1 for x in range(5)]
- evens = [number for number in range(50) if number % 2 == 0]
- max_heap = [(12, [0, 1])]
  [point for (dist, point) in max_heap]

## Dictionary comprehension

- pairs = [("a", 1), ("b", 2), ("c", 3)]
  my_dict = {k: v for k, v in pairs}

## Set comprehension

nums = [1, 1, 2]
unique_squares = {x \*\* 2 for x in nums}

## Generator comprehension

sum_of_squares = sum(x \*\* 2 for x in range(1000000))

## Conditional comprehension

- categories = ["Even" if x % 2 == 0 else "Odd" for x in range(10)]
- options = ["any", "w", "zh"]
  string_start_with_a_and_end_with_y = [
  string
  for string in options
  if len(string) >= 2
  if string[0] == "a"
  if string[-1] == "y"
  ]

## Function comprehension

def square(x):
return x \*\* 2

squared_numbers = [square(x) for x in range(10)]

## Nested comprehension

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened1 = [num for row in matrix for num in row]

# exterior for loop is first. then, interior for loops.

flattened2 = []
for row in matrix:
for num in row:
flattened2.append(num)

# Sorting

numbers = [3, 1, 4, 1, 5, 9, 2]
sorted_numbers = sorted(numbers, reverse=True)
numbers.sort(reverse=True)

# Custom sorting

```python
people = [('Alice', 30), ('Bob', 25), ('Charlie', 35)]
sorted_people = sorted(people, key=lambda person: person[1])  # Sort by age
people.sort(key=lambda person: person[1])  # Output will be sorted by age

people = [('Alice', 30), ('Bob', 25), ('Charlie', 30)]
sorted_people = sorted(people, key=lambda person: (person[1], person[0]))  # Sort by age, then by name
people.sort(key=lambda person: (person[1], person[0]))  # Sort by age, then by name

```

# Itertools

import itertools

- permutations = itertools.permutations(items, 2): Generate all possible permutations of length 2.
- combinations = itertools.combinations(items, 2): Generate all possible combinations of length 2.

# Math

- import math

sqrt_val = math.sqrt(x) # Return the square root of x
power_val = math.pow(x, y) # Return x raised to the power of y
floor_val = math.floor(3 / 2)
ceil_val = math.ceil(3 / 2)
sqrt_2 = math.sqrt(2)

# Float

float("inf") # Positive infinity
float("-inf") # Negative infinity

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

- Functions within functions don't need self syntax and can use variables from the outer function without needing to pass them

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

- Generator expressions: (x\*\*2 for x in range(10))
- def simple_gen():
  yield 1
  yield 2
  yield 3
  gen = simple_gen()
  print(next(gen)) # Output: 1
  print(next(gen)) # Output: 2
  print(next(gen)) # Output: 3

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

print(outer("a", "b")) # returns abc


# Can modify objects but not reassign them unless using the `nonlocal` keyword

def double(arr, val):
    def helper():
        # Modifying array works
        for i, n in enumerate(arr):
            arr[i] *= 2

        # This will only modify val in the helper scope
        # val *= 2
        # This will modify val outside the helper scope
        nonlocal val
        val *= 2

    helper()
    print(arr, val)


nums = [1, 2]
val = 3
double(nums, val) # prints [2, 4] 6. val became double because nonlocal scope was used. # nums array became double.
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


# Creating an instance of MyClass
my_obj = MyClass([1, 2, 3])
print(my_obj.get_length())  # Output: 3
print(my_obj.get_double_length())  # Output: 6
```

# Class Methods and Static Methods

class MyClass:
class_variable = "Class Variable"

    @classmethod
    def class_method(cls):
        return cls.class_variable

    @staticmethod
    def static_method(x, y):
        return x + y

# Class method example

print(MyClass.class_method()) # Output: Class Variable

# Static method example

print(MyClass.static_method(5, 3)) # Output: 8

# Inheritance

class BaseClass:
def **init**(self):
self.base_attr = "Base attribute"

    def base_method(self):
        return "Base method"

class DerivedClass(BaseClass):
def **init**(self):
super().**init**() # Initialize the parent class
self.derived_attr = "Derived attribute"

    def derived_method(self):
        return "Derived method"

# Creating an instance of DerivedClass

derived_obj = DerivedClass()
print(derived_obj.base_attr) # Output: Base attribute
print(derived_obj.derived_attr) # Output: Derived attribute
print(derived_obj.base_method()) # Output: Base method

# Syntax Essentials

## Multi-Assignment and Basic Operations

n, m, z = 0.125, "abc", False # multi-assignment
n += 1

if n > 2:
n -= 1
elif n == 2:
n \*= 2
else:
n += 2

## Loops and Iteration

# For loop with range

for i in range(5, 1, -1):
pass # Loop from 5 to 2

## Division and Modulo Operations

print(5 / 2) # Output: 2.5 (float division)
print(5 // 2) # Output: 2 (floor division)

# Handling negative numbers in division and modulo

print(-3 // 2) # Output: -2 -> most languages round towards 0 for negative. Python rounds towards negative infinity.
print(int(-3 / 2)) # Output: -1 -> Rounds towards zero
print(-10 % 3) # Output: 2 -> most languages round towards 0 for negative. Python moves away from 0.

import math
print(math.fmod(-10, 3)) # Output: -1.0 -> Correct modulo towards zero.

```


# Advanced Algorithms and Data Structures
## Graph Algorithms
### Graph Representation

# Adjacency List (commonly used for sparse graphs)
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

### Graph Traversal
# Depth-First Search (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for next in set(graph[start]) - visited:
        dfs(graph, next, visited)
    return visited

# Breadth-First Search (BFS)
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(set(graph[vertex]) - visited)
    return visited

# Topological Sort
# Kahn’s Algorithm for Topological Sorting
from collections import deque, defaultdict

def topological_sort_kahn(graph):
    in_degree = defaultdict(int)
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in graph if in_degree[node] == 0])
    top_order = []

    while queue:
        node = queue.popleft()
        top_order.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return top_order if len(top_order) == len(graph) else []

# Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Union-Find (Disjoint Set)
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

# Dynamic Programming Techniques

## Fibonacci Sequence Using Memoization
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]

# Problem-Solving Strategies
## Two-Pointer Technique

# Finding a Pair with a Given Sum in a Sorted Array
def two_sum(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1
        else:
            right -= 1

## Sliding Window
# Finding the Maximum Sum of k Consecutive Elements
def max_sum_subarray(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(len(nums) - k):
        window_sum = window_sum - nums[i] + nums[i + k]
        max_sum = max(max_sum, window_sum)
    return max_sum


## Backtracking
# Template for Backtracking
def backtrack(curr_state):
    if goal_reached(curr_state):
        return
    for choice in valid_choices:
        make_choice(curr_state, choice)
        backtrack(curr_state)
        undo_choice(curr_state, choice)

## Bit Manipulation
# Common Bitwise Operations
even_or_odd = n & 1  # Check if a number is even/odd
ith_bit = (n >> i) & 1  # Get the ith bit
n = n | (1 << i)  # Set the ith bit
n = n ^ (1 << i)  # Flip the ith bit
n = n & (n - 1)  # Remove the last set bit
is_power_of_two = n > 0 and (n & (n - 1)) == 0  # Check if a number is a power of two

# Final Tips
# Understanding the Problem
# Always clarify the problem statement and understand the input-output requirements before coding.

# Edge Cases
# Consider edge cases like empty inputs, large inputs, negative values, etc.

# Efficiency
# Focus on optimizing time and space complexity, especially for large inputs.

# Practice
# Regularly solve problems on platforms like LeetCode, HackerRank, or Codeforces to build and maintain problem-solving skills.
```
