from langchain_core.tools import tool

@tool
def sqaure(a: int) -> int:
    """Sqaure of the number"""
    return a**2

@tool
def factorial(a: int) -> int:
    """Factorial of the number"""
    fact = 1
    for i in range(1, a+1):
        fact*=i
    return fact


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

print(factorial.invoke({'a': 4}))
print(sqaure.invoke({'a': 4}))
print(multiply.invoke({'a': 4, 'b': 10}))
