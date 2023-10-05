
class UserInputManager:
    """
    The UserInputManager class provides static methods to handle user inputs.
    """

    @staticmethod
    def query_yes_no(prompt):
        """
        Prompts the user with a yes/no question and returns a boolean value based on the user's response.

        Args:
            prompt (str): The question to be displayed to the user.

        Returns:
            bool: True if the user's response is 'yes', 'y', or '1'. False if the response is 'no', 'n', or '0'.
        """
        while True:
            response = input(prompt + " (yes/no): ").strip().lower()
            if response in ["yes", "y", "1"]:
                return True
            elif response in ["no", "n", "0"]:
                return False
            else:
                print("Invalid input. Please enter yes/no, y/n, 1/0.")

    @staticmethod
    def get_action():
        """
        Prompts the user to choose between training or evaluating and returns the user's choice.

        Returns:
            str: 't' for training or 'e' for evaluating.
        """
        action = input("Do you want to train or evaluate? (t/e): ").strip().lower()
        while action not in ["t", "e"]:
            print("Invalid input. Please enter t or e.")
            action = input("Do you want to train or evaluate? (t/e): ").strip().lower()
        return action

