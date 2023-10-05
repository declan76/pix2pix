from managers.train_manager import TrainingManager
from managers.user_input_manager import UserInputManager
from managers.evaluation_manager import EvaluationManager

"""
Main entry point for the application.
"""
def main():
    action = UserInputManager.get_action()

    if action == "t":
        training_manager = TrainingManager()
        training_manager.orchestrate_training()
    elif action == "e":
        evaluator_manager = EvaluationManager()
        evaluator_manager.orchestrate_evaluation()
        

if __name__ == "__main__":
    main()
