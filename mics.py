import os
import shutil
from git import Repo

def copy_directory_from_git(repo_url, branch, directory_to_copy, destination_path):
    """
    Clone a specific directory from a Git repository.

    :param repo_url: URL of the Git repository.
    :param branch: The branch to clone (e.g., 'main' or 'master').
    :param directory_to_copy: Path of the directory inside the repo to copy.
    :param destination_path: Path to copy the directory to.
    """
    # Create a temporary directory to clone the repo
    temp_repo_path = "temp_repo"
    
    try:
        print(f"Cloning {repo_url}...")
        Repo.clone_from(repo_url, temp_repo_path, branch=branch, depth=1)
        
        # Construct the full path to the directory in the repo
        source_path = os.path.join(temp_repo_path, directory_to_copy)
        
        # Ensure the source directory exists
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Directory {directory_to_copy} not found in repository.")
        
        # Copy the directory to the destination path
        print(f"Copying {directory_to_copy} to {destination_path}...")
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        
        print("Directory copied successfully.")
    
    finally:
        # Cleanup: Remove the cloned repository
        if os.path.exists(temp_repo_path):
            shutil.rmtree(temp_repo_path)
            print("Temporary files cleaned up.")

# Example usage
repo_url = "https://github.com/your-username/your-repo.git"
branch = "main"
directory_to_copy = "path/to/directory/in/repo"
destination_path = "path/to/destination"

copy_directory_from_git(repo_url, branch, directory_to_copy, destination_path)
