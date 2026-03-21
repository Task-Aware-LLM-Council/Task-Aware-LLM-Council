from huggingface_hub import login, upload_folder
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--folder-path", required=True,
                        help="Path to the dataset folder")
    parser.add_argument("--repo-id", required=False,
                        help="Hugging Face repository ID", default="mbhaskar98/router-dataset")
    parser.add_argument("--commit-message", required=True,
                        help="Commit message for the upload")

    args = parser.parse_args()

    login()

    print(f"Uploading dataset from {args.folder_path} to Hugging Face Hub repository {args.repo_id}...")

    upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message
    )
