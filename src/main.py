from rock_paper_scissors import *

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