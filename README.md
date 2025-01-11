This a Python program that simulates a game of Rock, Paper, Scissors between the user and the computer.

The code for the Game is developed in Jupyther Notebook and Python script that can be run on the terminal using this line of code : "python ./rock_paper_scissors/main.py "

Explanation
Define the Possible Choices:

choices = ["Rock", "Paper", "Scissors"]: Defines a list of possible choices.
Get the User's Choice:

get_user_choice(): Prompts the user to enter their choice and ensures it is valid by checking against the choices list. If the input is invalid, it prompts the user again.
Generate a Random Choice for the Computer:

get_computer_choice(): Uses random.choice(choices) to generate a random choice for the computer.
Determine the Winner:

determine_winner(user_choice, computer_choice): Implements the rules of Rock, Paper, Scissors to determine the winner. It checks for ties and determines the winner based on the rules.
Main Game Loop:

play_game(): Orchestrates the game by calling the functions to get the user's choice, computer's choice, and determine the winner. It then prints the results.
Running the Code
To run the code in a Jupyter Notebook, simply copy and paste the entire code block into a cell and execute it. Here’s how it looks in a Jupyter Notebook:

Version Control: Use Git for version control. Initialize a Git repository in your project directory:
git init
git add .
git commit -m "Initial commit"

Virtual Environments: Use virtual environments to manage dependencies:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt