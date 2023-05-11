

class CustomStack:

    def __init__(self, size):
        self.stack = []
        self.length = 0
        self.size = size

    def push(self, item):
        if self.length == self.size:
            self.pop()
        self.stack.append(item)
        self.length += 1

    def pop(self):
        if self.length == 0:
            raise IndexError("pop from an empty stack")
        self.length -= 1
        item = self.stack[0]
        self.stack.pop(0)
        return item

    def get(self, index):
        if index >= self.length:
            raise IndexError(f"Index {index} out of range {self.length}")
        return self.stack[index]

    def __len__(self):
        return self.length

    def __str__(self):
        return str(self.stack[:self.length])
