'''
This is a version of the Python Progagramme implementation of the Rock Paper Scissors game
'''
# Step 1: Define the possible choices
choices = ["Rock", "Paper", "Scissors"]

# Step 2: Get the user's choice
def get_user_choice():
    """
    Prompts the user to enter their choice of Rock, Paper, or Scissors.
    
    Returns:
        str: The user's choice, capitalized and validated.
    """
    user_input = input("Enter your choice (Rock, Paper, Scissors): ").strip().capitalize()
    while user_input not in choices:
        print("Invalid choice. Please try again.")
        user_input = input("Enter your choice (Rock, Paper, Scissors): ").strip().capitalize()
    return user_input

# Step 3: Generate a random choice for the computer
import random

def get_computer_choice():
    """
    Generates a random choice of Rock, Paper, or Scissors for the computer.
    
    Returns:
        str: The computer's choice.
    """
    return random.choice(choices)

# Step 4: Determine the winner
def determine_winner(user_choice, computer_choice):
    """
    Determines the winner of the Rock, Paper, Scissors game based on the user's and computer's choices.
    
    Parameters:
        user_choice (str): The user's choice (Rock, Paper, or Scissors).
        computer_choice (str): The computer's choice (Rock, Paper, or Scissors).
    
    Returns:
        str: The result of the game (e.g., "You win!", "Computer wins!", "It's a tie!").
    """
    if user_choice == computer_choice:
        return "It's a tie!"
    elif (user_choice == "Rock" and computer_choice == "Scissors") or \
         (user_choice == "Scissors" and computer_choice == "Paper") or \
         (user_choice == "Paper" and computer_choice == "Rock"):
        return "You win!"
    else:
        return "Computer wins!"

# Main game loop
def play_game():
    """
    Orchestrates the Rock, Paper, Scissors game by getting the user's choice,
    generating a random choice for the computer, and determining the winner.
    """
    print("Welcome to Rock, Paper, Scissors!")
    user_choice = get_user_choice()
    computer_choice = get_computer_choice()
    
    print(f"You chose: {user_choice}")
    print(f"Computer chose: {computer_choice}")
    
    result = determine_winner(user_choice, computer_choice)
    print(result)

# Run the game
if __name__ == "__main__":
    play_game()