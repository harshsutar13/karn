"""
Simple Calculator in Python
Performs basic arithmetic operations: addition, subtraction, multiplication, division
"""

def add(x, y):
    """Add two numbers"""
    return x + y

def subtract(x, y):
    """Subtract two numbers"""
    return x - y

def multiply(x, y):
    """Multiply two numbers"""
    return x * y

def divide(x, y):
    """Divide two numbers"""
    if y == 0:
        return "Error! Division by zero."
    return x / y

def calculator():
    """Main calculator function"""
    print("=" * 40)
    print("Simple Calculator")
    print("=" * 40)
    print("\nSelect operation:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")

    while True:
        choice = input("\nEnter choice (1/2/3/4) or 'q' to quit: ")

        if choice.lower() == 'q':
            print("Thank you for using the calculator!")
            break

        if choice in ['1', '2', '3', '4']:
            try:
                num1 = float(input("Enter first number: "))
                num2 = float(input("Enter second number: "))

                if choice == '1':
                    result = add(num1, num2)
                    operation = "+"
                elif choice == '2':
                    result = subtract(num1, num2)
                    operation = "-"
                elif choice == '3':
                    result = multiply(num1, num2)
                    operation = "*"
                elif choice == '4':
                    result = divide(num1, num2)
                    operation = "/"

                print(f"\n{num1} {operation} {num2} = {result}")

            except ValueError:
                print("Invalid input! Please enter valid numbers.")
        else:
            print("Invalid choice! Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    calculator()
