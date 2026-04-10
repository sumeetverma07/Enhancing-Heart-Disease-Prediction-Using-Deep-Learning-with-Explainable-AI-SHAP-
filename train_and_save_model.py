from pathlib import Path

from model import train_and_save_pipeline


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    artifacts = train_and_save_pipeline(project_dir=project_dir, force_retrain=True)
    print("Training completed.")
    print(artifacts["comparison_df"])
