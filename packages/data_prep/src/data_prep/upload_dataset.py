from huggingface_hub import login, upload_folder, create_repo
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--folder-path", required=True,
                        help="Path to the dataset folder")
    parser.add_argument("--repo-id", required=False,
                        help="Hugging Face repository ID", default="task-aware-llm-council/router_dataset-2")
    parser.add_argument("--commit-message", required=True,
                        help="Commit message for the upload")

    args = parser.parse_args()

    login()

    print(f"Uploading dataset from {args.folder_path} to Hugging Face Hub repository {args.repo_id}...")

    # token = "YOUR_HF_TOKEN"

    # Create the dataset repo if it doesn't exist
    create_repo(
        repo_id=args.repo_id,
        # token=token,
        repo_type="dataset",  # important for datasets
        private=False,         # set True if you want private
        exist_ok=True  # <-- skip if already exists
    )

    upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message
    )
