from managers.train_manager import TrainingManager
from managers.user_input_manager import UserInputManager
from managers.evaluation_manager import EvaluationManager

class App:
    """
    A class to handle the main operations of the application.
    
    This class serves as the main entry point for the application, 
    orchestrating the training and evaluation processes based on user input.
    """

    def __init__(self):
        pass

    def run(self):
        """
        Main method to execute the application's primary logic.
        
        This method retrieves the user's desired action (training or evaluation)
        and then invokes the appropriate manager to handle the selected action.
        """
        action = UserInputManager.get_action()

        if action == "t":
            training_manager = TrainingManager()
            training_manager.orchestrate_training()
        elif action == "e":
            evaluator_manager = EvaluationManager()
            evaluator_manager.orchestrate_evaluation()

if __name__ == "__main__":
    application = App()
    application.run()
